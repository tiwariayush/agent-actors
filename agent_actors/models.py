from typing import List

from pydantic import BaseModel, Field, root_validator


def _default_missing_task_id(values):
    """Fill missing/null task_id with 0 when child_id is already set.

    The Plan prompt numbers tasks as an "incrementing int starting at 0 per
    child". Models often omit task_id when assigning a single task per worker,
    or copy the unquoted ``<incrementing int...>`` placeholder as JSON null,
    while still setting child_id. That is valid JSON, but TaskRef requires
    task_id, so ParentAgent.run raises ValidationError before any child work
    starts. The same crash happens for dependency objects that only name
    child_id.

    Leave mappings that already have an ``id`` field untouched so combined-id
    and numeric/quoted ``id`` repairs remain independent. Do not invent a
    child_id — a missing worker cannot be assigned safely.
    """
    if not isinstance(values, dict):
        return values
    if values.get("task_id") is not None:
        return values
    if values.get("child_id") is None:
        return values
    if "id" in values:
        return values

    promoted = dict(values)
    promoted["task_id"] = 0
    return promoted


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _fill_missing_task_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # sets child_id but omits task_id (or emits JSON null for it).
        return _default_missing_task_id(values)

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
