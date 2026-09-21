from typing import List

from pydantic import BaseModel, Field, root_validator

# Hyphenated task-text key copied from the Plan prompt's "sub-task" wording.
# Distinct from draft PR #110 (sub_task / subtask) and from #112 (plan-level
# {sub-tasks|sub-task} wrappers).
_HYPHENATED_SUB_TASK_ALIASES = ("sub-task",)


def _promote_hyphenated_sub_task(values):
    """Fill missing task from hyphenated sub-task.

    The Plan prompt asks for a JSON ``task`` field but describes the work as
    hyphenated sub-tasks throughout. Models often emit ``"sub-task"`` instead.
    Without this, ParentAgent.run raises ValidationError after json.loads
    succeeds.

    Leave mappings that already have a task key untouched so JSON-null,
    list-valued, and object-shaped task repairs remain independent. Leave
    underscore sub_task/subtask for draft PR #110. Leave description/text/
    objective/instruction siblings for the missing-task repair. Leave
    plan-level ``{"sub-tasks"|"sub-task": [...]}`` wrappers for draft PR #112.
    """
    if not isinstance(values, dict):
        return values
    if "task" in values:
        return values
    for key in _HYPHENATED_SUB_TASK_ALIASES:
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
    def _alias_hyphenated_sub_task(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses "sub-task" (from the prompt's hyphenated wording) and omits task.
        return _promote_hyphenated_sub_task(values)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
