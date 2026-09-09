from typing import List

from pydantic import BaseModel, Field, validator


def _default_null_task(value):
    """Replace JSON-null task with an empty string.

    The Plan prompt shows ``"task": <task task>``. JSON-aware models copy
    that unquoted placeholder as JSON null while still setting child_id
    and task_id. That is valid JSON, but TaskRecord requires a string, so
    ParentAgent.run raises ValidationError before any child work starts.

    List-valued tasks, object-shaped tasks, and a missing task key stay
    untouched so other draft repairs remain independent.
    """
    if value is None:
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
    def _coerce_null_task(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON uses JSON
        # null for the task field after json.loads succeeds.
        return _default_null_task(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
