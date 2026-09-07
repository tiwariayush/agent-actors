from typing import List

from pydantic import BaseModel, Field, root_validator


def _promote_digit_string_id(values):
    """Fill missing task_id from a digit-only string `id` when child_id is set.

    The Plan prompt identifies work as [worker #.task #]. Models often emit
    child_id correctly and then quote the task index as a generic `id`
    field (`"id": "0"`). That is valid JSON, but `id` is only a property on
    TaskRef, so Pydantic ignores it and ParentAgent.run raises
    ValidationError for the missing task_id before any child work starts.
    """
    if not isinstance(values, dict) or "id" not in values:
        return values
    if values.get("task_id") is not None or values.get("child_id") is None:
        return values

    raw = values["id"]
    if not isinstance(raw, str):
        return values
    stripped = raw.strip()
    if not stripped.isdigit():
        return values

    promoted = dict(values)
    promoted.pop("id", None)
    promoted["task_id"] = int(stripped)
    return promoted


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _alias_digit_string_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # quotes a task index as `"id": "0"` while already setting child_id.
        return _promote_digit_string_id(values)

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
