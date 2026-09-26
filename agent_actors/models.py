from typing import List

from pydantic import BaseModel, Field, root_validator

# Key a Plan model uses instead of task_id after reading the prompt's
# plural hyphenated "sub-tasks" wording next to the schema's task_id.
# Distinct from draft PR #114 (singular hyphenated sub-task_id),
# #111 (underscore sub_task_id / subtask_id), and #115 (plural
# "sub-tasks" as the task *text*).
_PLURAL_SUB_TASKS_ID = "sub-tasks_id"


def _promote_plural_sub_tasks_id(values):
    """Fill missing task_id from plural hyphenated ``sub-tasks_id``.

    The Plan prompt asks for a JSON ``task_id`` field but describes the
    work as "sub-tasks" four times. Models that copy that plural spelling
    into the key name emit ``sub-tasks_id`` instead of ``task_id``.
    Without this, ParentAgent.run raises ValidationError after json.loads
    succeeds.

    Leave mappings that already have task_id untouched. Leave mappings
    that already have an ``id`` field so combined-id repairs (#93/#94/#99)
    stay independent. Leave singular ``sub-task_id`` so #114 stays
    independent, and underscore ``sub_task_id`` / ``subtask_id`` so #111
    stays independent. Leave a missing task_id without this alias so the
    default-to-0 repair (#100) stays independent. Leave fully hyphenated
    ``sub-task-id`` and plural task text ``sub-tasks`` so those repairs
    stay independent.
    """
    if not isinstance(values, dict):
        return values
    if values.get("task_id") is not None:
        return values
    if "id" in values:
        return values
    if _PLURAL_SUB_TASKS_ID not in values:
        return values
    raw = values[_PLURAL_SUB_TASKS_ID]
    if isinstance(raw, bool) or raw is None:
        return values
    promoted = dict(values)
    promoted["task_id"] = promoted.pop(_PLURAL_SUB_TASKS_ID)
    return promoted


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _alias_plural_sub_tasks_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses plural sub-tasks_id (from the prompt's "sub-tasks" wording)
        # and omits task_id on a task or dependency object.
        return _promote_plural_sub_tasks_id(values)

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
