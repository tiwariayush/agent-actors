import re
from typing import List

from pydantic import BaseModel, Field, validator

# Plan prompt asks models to reference tasks as [worker #.task #]; LLMs often
# put that string form (or the bare "child.task" id) into dependencies.
_TASK_REF_STRING = re.compile(
    r"^(?:\[(?P<bracket_child>\d+)\.(?P<bracket_task>\d+)\]|"
    r"(?P<bare_child>\d+)\.(?P<bare_task>\d+))$"
)


def _coerce_task_ref(value):
    """Normalize a single dependency entry into a TaskRef-shaped mapping."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        match = _TASK_REF_STRING.fullmatch(value.strip())
        if match:
            child_id = match.group("bracket_child") or match.group("bare_child")
            task_id = match.group("bracket_task") or match.group("bare_task")
            return {"child_id": int(child_id), "task_id": int(task_id)}
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
        # Plan LLMs often emit dependency references as strings matching the
        # prompt's [worker #.task #] form (or bare "child.task" ids). Without
        # coercion, ParentAgent.run crashes during TaskRecord construction.
        if isinstance(value, str):
            return [_coerce_task_ref(value)]
        if isinstance(value, list):
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
