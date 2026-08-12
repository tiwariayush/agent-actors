from typing import List

from pydantic import BaseModel, Field, validator


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @validator("dependencies", pre=True)
    def _normalize_dependencies(cls, value):
        # Plan LLMs often emit an empty object for "no dependencies" (the
        # zero-dependency analogue of a bare single dependency object).
        # Without coercion, ParentAgent.run crashes while building TaskRecords.
        # Distinct from draft PR #71 (null), #75 (non-empty object wrap), and
        # #79 (null/empty placeholders inside an array).
        if isinstance(value, dict) and len(value) == 0:
            return []
        return value

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
