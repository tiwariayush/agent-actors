from typing import List

from pydantic import BaseModel, Field, validator


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


def _is_null_dependency_placeholder(item) -> bool:
    """True for null entries or objects with both task ref ids unset."""
    if item is None:
        return True
    if isinstance(item, dict):
        return item.get("child_id") is None and item.get("task_id") is None
    return False


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @validator("dependencies", pre=True)
    def _normalize_dependencies(cls, value):
        # The Plan prompt always shows dependencies as an array of objects.
        # For tasks with no prerequisites, models often keep that shape but null
        # the placeholder (e.g. [null] or [{"child_id": null, "task_id": null}]).
        # Pydantic then raises ValidationError and ParentAgent.run aborts before
        # dispatch. Distinct from top-level "dependencies": null (draft PR #71).
        if isinstance(value, list):
            return [
                item for item in value if not _is_null_dependency_placeholder(item)
            ]
        return value

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
