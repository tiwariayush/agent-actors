import re
from typing import List

from pydantic import BaseModel, Field, root_validator

# Plan prompt identifies tasks as [worker #.task #]; models often emit that
# as a single `id` field instead of child_id / task_id.
_COMBINED_ID = re.compile(r"^\[?(?P<child>\d+)\.(?P<task>\d+)\]?$")


def _promote_combined_id(values):
    """Fill missing child_id/task_id from a dotted `id` (`0.0` / `[0.0]`)."""
    if not isinstance(values, dict) or "id" not in values:
        return values
    if values.get("child_id") is not None and values.get("task_id") is not None:
        return values
    combined = values["id"]
    if not isinstance(combined, str):
        return values
    match = _COMBINED_ID.fullmatch(combined.strip())
    if not match:
        return values
    promoted = dict(values)
    promoted.pop("id", None)
    if promoted.get("child_id") is None:
        promoted["child_id"] = int(match.group("child"))
    if promoted.get("task_id") is None:
        promoted["task_id"] = int(match.group("task"))
    return promoted


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _split_combined_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses the prompt's [worker #.task #] form as `id` and omits
        # child_id / task_id on a task or dependency object.
        return _promote_combined_id(values)

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
