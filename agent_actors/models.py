from typing import List

from pydantic import BaseModel, Field

# Plan-level wrappers copied from the prompt's hyphenated "sub-task" /
# "sub-tasks" wording. Distinct from draft PR #74 (subtasks / sub_tasks /
# tasks / plan) and from #110 (sub_task / subtask as the task *text*).
_HYPHENATED_SUB_TASK_PLAN_KEYS = ("sub-tasks", "sub-task")

# Canonical task-object fields on main. A mapping that already has these
# is a task, not a plan wrapper, even if it also has a hyphenated key.
_TASK_OBJECT_MARKER_KEYS = frozenset(
    {"task", "task_id", "child_id", "id", "dependencies"}
)


def unwrap_hyphenated_sub_task_plan(raw_plan):
    """Unwrap a hyphenated sub-task(s) Plan wrapper into a task list.

    The Plan prompt asks for a JSON array but describes the work as
    "sub-tasks" / "the sub-task". Models that copy that spelling wrap the
    array as ``{"sub-tasks": [...]}`` or ``{"sub-task": [...]}``. Iterating
    the wrapper yields those string keys, and ``TaskRecord(**key)`` raises
    ``TypeError``, aborting the parent run before any child work starts.

    Leave lists, ``{"tasks"|"subtasks"|"sub_tasks"|"plan": ...}`` wrappers
    (#74), numeric/dotted-keyed objects (#92/#95), and bare task objects
    (#73) unchanged. Leave a task object that uses ``sub-task`` as a text
    field so a field-alias repair stays independent of this wrapper unwrap.
    """
    if not isinstance(raw_plan, dict) or not raw_plan:
        return raw_plan
    if _TASK_OBJECT_MARKER_KEYS.intersection(raw_plan):
        return raw_plan
    for key in _HYPHENATED_SUB_TASK_PLAN_KEYS:
        if key not in raw_plan:
            continue
        value = raw_plan[key]
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return raw_plan
    return raw_plan


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

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
