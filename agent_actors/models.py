from typing import List

from pydantic import BaseModel, Field, root_validator

# Plural hyphenated task-text key copied from the Plan prompt's "sub-tasks"
# wording. Distinct from draft PR #113 (singular "sub-task") and from #112
# (plan-level {"sub-tasks"|"sub-task": [...]} wrappers).
_PLURAL_HYPHENATED_SUB_TASKS = "sub-tasks"


def _promote_plural_hyphenated_sub_tasks(values):
    """Fill missing task from plural hyphenated ``sub-tasks``.

    The Plan prompt asks for a JSON ``task`` field but names the work
    "sub-tasks" throughout ("new sub-tasks", "Assign sub-tasks", "multiple
    sub-tasks", "existing sub-tasks"). Models that copy that spelling into
    the key name emit a string ``sub-tasks`` instead of ``task``. Without
    this, ParentAgent.run raises ValidationError after json.loads succeeds.

    Leave mappings that already have a task key untouched so JSON-null,
    list-valued, and object-shaped task repairs remain independent. Leave
    singular ``sub-task`` for draft PR #113, underscore sub_task/subtask for
    #110, and description/text/objective/instruction siblings for the
    missing-task repair. Leave list and object ``sub-tasks`` values so
    plan-level wrappers (#112) stay independent.
    """
    if not isinstance(values, dict):
        return values
    if "task" in values:
        return values
    if _PLURAL_HYPHENATED_SUB_TASKS not in values:
        return values
    raw = values[_PLURAL_HYPHENATED_SUB_TASKS]
    if not isinstance(raw, str) or not raw:
        return values
    promoted = dict(values)
    promoted["task"] = promoted.pop(_PLURAL_HYPHENATED_SUB_TASKS)
    return promoted


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
    def _alias_plural_hyphenated_sub_tasks(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses "sub-tasks" (the prompt's plural hyphenated wording) and omits task.
        return _promote_plural_hyphenated_sub_tasks(values)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
