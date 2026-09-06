from typing import List

from pydantic import BaseModel, Field, validator


def _join_list_task(value):
    """Join a non-empty list of task step strings into one task string.

    Plan models sometimes emit ``"task": ["step a", "step b"]`` for a
    multi-part objective. That is valid JSON, but TaskRecord requires a
    string, so ParentAgent.run raises ValidationError before any child
    work starts. Non-string items, empty lists, and null stay untouched
    so other draft repairs remain independent.
    """
    if isinstance(value, list) and value and all(
        isinstance(item, str) for item in value
    ):
        return "\n".join(item.strip() for item in value)
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
    def _coerce_list_task(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON uses an
        # array of steps for the task field after json.loads succeeds.
        return _join_list_task(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
