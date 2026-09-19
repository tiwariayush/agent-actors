from typing import List

from pydantic import BaseModel, Field, root_validator

# Keys a Plan model uses instead of task_id after reading the prompt's
# repeated "sub-task" / "sub-tasks" wording next to the schema's task_id.
# Distinct from draft PR #110 (sub_task / subtask aliases for the task
# *text*), and from #100 (missing/null task_id defaulting to 0).
_SUB_TASK_ID_ALIASES = ("sub_task_id", "subtask_id")


def _promote_sub_task_id(values):
    """Fill missing task_id from sub_task_id or subtask_id.

    The Plan prompt asks for a JSON ``task_id`` field but describes the
    work as sub-tasks throughout. Models that copy that wording into the
    key name emit sub_task_id or subtask_id instead. Without this,
    ParentAgent.run raises ValidationError after json.loads succeeds.

    Leave mappings that already have task_id untouched. Leave mappings
    that already have an ``id`` field so combined-id repairs (#93/#94/#99)
    stay independent. Leave a missing task_id without an alias so the
    default-to-0 repair (#100) stays independent.
    """
    if not isinstance(values, dict):
        return values
    if values.get("task_id") is not None:
        return values
    if "id" in values:
        return values
    for key in _SUB_TASK_ID_ALIASES:
        if key not in values:
            continue
        raw = values[key]
        if isinstance(raw, bool) or raw is None:
            return values
        promoted = dict(values)
        promoted["task_id"] = promoted.pop(key)
        return promoted
    return values


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _alias_sub_task_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses sub_task_id / subtask_id (from the prompt's sub-task wording)
        # and omits task_id on a task or dependency object.
        return _promote_sub_task_id(values)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
