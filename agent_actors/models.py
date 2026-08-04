import re
from typing import List

from pydantic import BaseModel, Field, validator

# Plan prompt tells models to cite tasks as [worker #.task #]. In JSON that
# often becomes a two-element array [worker, task] (or the string "[0, 0]"),
# rather than {"child_id": ..., "task_id": ...}.
_TASK_REF_PAIR_STRING = re.compile(r"^\[(\d+)\s*,\s*(\d+)\]$")


def _is_int_like(value) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (
        isinstance(value, float) and value.is_integer()
    )


def _coerce_task_ref(value):
    """Normalize one dependency entry into a TaskRef-shaped mapping when possible."""
    if isinstance(value, dict):
        return value

    if isinstance(value, (list, tuple)) and len(value) == 2:
        child_id, task_id = value
        if _is_int_like(child_id) and _is_int_like(task_id):
            return {"child_id": int(child_id), "task_id": int(task_id)}
        return value

    if isinstance(value, str):
        match = _TASK_REF_PAIR_STRING.fullmatch(value.strip())
        if match:
            return {
                "child_id": int(match.group(1)),
                "task_id": int(match.group(2)),
            }

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

    @validator("dependencies", pre=True)
    def _normalize_dependencies(cls, value):
        # Prompt-induced pair forms: "dependencies": [0, 0], [[0, 0]], or "[0, 0]".
        # Without coercion, ParentAgent.run crashes while building TaskRecords.
        if isinstance(value, str):
            return [_coerce_task_ref(value)]

        if isinstance(value, (list, tuple)):
            if len(value) == 2 and all(_is_int_like(item) for item in value):
                return [_coerce_task_ref(value)]
            return [_coerce_task_ref(item) for item in value]

        return value

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
