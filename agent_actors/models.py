from typing import List

from pydantic import BaseModel, Field, validator


def _default_empty_list_task(value):
    """Replace an empty-list task with an empty string.

    Plan models sometimes emit ``"task": []`` for a child that has ids but
    no steps yet — the zero-step form of a list-valued objective. That is
    valid JSON, but TaskRecord requires a str, so ParentAgent.run raises
    ValidationError before any child work starts.

    Non-empty string lists, mixed/object lists, JSON-null tasks, object
    tasks, and a missing task key stay untouched so those draft repairs
    remain independent.
    """
    if isinstance(value, list) and not value:
        return ""
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
    def _coerce_empty_list_task(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON uses an
        # empty array for the task field after json.loads succeeds.
        return _default_empty_list_task(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
