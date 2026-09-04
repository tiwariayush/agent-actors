from typing import List

from pydantic import BaseModel, Field, root_validator


def _promote_worker_id(values):
    """Fill missing child_id from worker_id.

    The Plan prompt identifies work as [worker #.task #]. Models often emit
    worker_id instead of child_id on a task or dependency object. Without this,
    ParentAgent.run raises ValidationError after json.loads succeeds.
    """
    if not isinstance(values, dict):
        return values
    if values.get("child_id") is not None:
        return values
    if "worker_id" not in values:
        return values
    raw = values["worker_id"]
    if isinstance(raw, bool) or raw is None:
        return values
    promoted = dict(values)
    promoted["child_id"] = promoted.pop("worker_id")
    return promoted


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _alias_worker_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses worker_id (from the prompt's [worker #.task #] form) and omits
        # child_id on a task or dependency object.
        return _promote_worker_id(values)

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
