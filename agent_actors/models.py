from typing import List

from pydantic import BaseModel, Field, root_validator


def _promote_numeric_id(values):
    """Fill missing child_id/task_id from a numeric `id`.

    The Plan prompt identifies work as [worker #.task #]. Models often put
    that citation in a single `id` field. Unquoted JSON `0.0` / `1.2` becomes
    a float and encodes both halves; a bare integer `id` is a task_id alias
    when child_id is already present.
    """
    if not isinstance(values, dict) or "id" not in values:
        return values
    if values.get("child_id") is not None and values.get("task_id") is not None:
        return values

    raw = values["id"]
    if isinstance(raw, bool):
        return values

    if isinstance(raw, float):
        parts = str(raw).split(".")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return values
        promoted = dict(values)
        promoted.pop("id", None)
        if promoted.get("child_id") is None:
            promoted["child_id"] = int(parts[0])
        if promoted.get("task_id") is None:
            promoted["task_id"] = int(parts[1])
        return promoted

    if isinstance(raw, int):
        if values.get("task_id") is not None or values.get("child_id") is None:
            return values
        promoted = dict(values)
        promoted.pop("id", None)
        promoted["task_id"] = raw
        return promoted

    return values


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _split_numeric_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses an unquoted numeric `id` (`0.0` or `0`) instead of child_id /
        # task_id on a task or dependency object.
        return _promote_numeric_id(values)

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
