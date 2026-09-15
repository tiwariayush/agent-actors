from typing import List

from pydantic import BaseModel, Field, validator

# Object-task keys already handled by the known-key extractor (draft PR #102).
_KNOWN_OBJECT_TASK_KEYS = (
    "task",
    "description",
    "text",
    "objective",
    "instruction",
)


def _task_from_empty_or_unknown_object(value):
    """Coerce empty or unknown-key object-shaped Plan task fields.

    Plan models sometimes emit ``"task": {}`` or a single unknown string
    field such as ``"task": {"title": "..."}`` / ``{"notes": "..."}``
    instead of a plain string. That is valid JSON, but TaskRecord requires
    a str, so ParentAgent.run raises ValidationError before any child work
    starts.

    Objects that already use the known keys from draft PR #102, JSON-null
    tasks, list-valued tasks, and a missing task key stay untouched.
    """
    if not isinstance(value, dict):
        return value
    if any(key in value for key in _KNOWN_OBJECT_TASK_KEYS):
        return value
    if not value:
        return ""
    strings = [item for item in value.values() if isinstance(item, str)]
    if len(strings) == 1:
        return strings[0]
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
    def _coerce_empty_or_unknown_object_task(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON uses an
        # empty or unknown-key object for the task field after json.loads
        # succeeds.
        return _task_from_empty_or_unknown_object(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
