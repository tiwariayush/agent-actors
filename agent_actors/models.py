from typing import List

from pydantic import BaseModel, Field, root_validator

# Keys a Plan model uses instead of task after reading the prompt's
# repeated "sub-task" / "sub-tasks" wording. Distinct from the plan-level
# {subtasks|sub_tasks} wrappers in draft PR #74, and from sibling
# description/text/objective/instruction copies in draft PR #103.
_SUB_TASK_ALIASES = ("sub_task", "subtask")


def _promote_sub_task(values):
    """Fill missing task from sub_task or subtask.

    The Plan prompt asks for a JSON ``task`` field but describes the work as
    sub-tasks throughout. Models often emit sub_task or subtask instead.
    Without this, ParentAgent.run raises ValidationError after json.loads
    succeeds.

    Leave mappings that already have a task key untouched so JSON-null,
    list-valued, and object-shaped task repairs remain independent. Leave
    description/text/objective/instruction siblings for the missing-task
    repair.
    """
    if not isinstance(values, dict):
        return values
    if "task" in values:
        return values
    for key in _SUB_TASK_ALIASES:
        if key not in values:
            continue
        raw = values[key]
        if not isinstance(raw, str) or not raw:
            continue
        promoted = dict(values)
        promoted["task"] = promoted.pop(key)
        return promoted
    return values


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @root_validator(pre=True)
    def _alias_sub_task(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses sub_task / subtask (from the prompt's sub-task wording) and
        # omits task.
        return _promote_sub_task(values)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
