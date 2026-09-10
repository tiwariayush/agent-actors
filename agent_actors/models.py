from typing import List

from pydantic import BaseModel, Field, validator

# Keys a Plan model uses when it nests the objective inside a task object.
_TASK_OBJECT_STRING_KEYS = (
    "task",
    "description",
    "text",
    "objective",
    "instruction",
)


def _task_from_object(value):
    """Extract a string from an object-shaped Plan task field.

    Plan models sometimes emit ``"task": {"description": "..."}`` or a
    nested ``{"task": "..."}`` instead of a plain string. That is valid
    JSON, but TaskRecord requires a str, so ParentAgent.run raises
    ValidationError before any child work starts.

    Lists, null, empty objects, and a missing task key stay untouched so
    other draft repairs remain independent.
    """
    if not isinstance(value, dict) or not value:
        return value
    for key in _TASK_OBJECT_STRING_KEYS:
        item = value.get(key)
        if isinstance(item, str):
            return item
    return value


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @validator("task", pre=True)
    def _coerce_object_task(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON uses an
        # object for the task field after json.loads succeeds.
        return _task_from_object(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
