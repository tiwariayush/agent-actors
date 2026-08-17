from typing import List

from pydantic import BaseModel, Field, validator


def _coerce_dotted_float_ref(value):
    """Normalize a JSON number like 0.0 / 0.1 into a TaskRef mapping.

    The Plan prompt tells models to cite tasks as [worker #.task #]. In JSON
    that citation is often an unquoted number (`0.0`, `0.1`) rather than a
    string or a {child_id, task_id} object. json.loads turns those into
    floats, and Pydantic then rejects them as TaskRef values.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, float) and not isinstance(value, bool):
        parts = str(value).split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return {"child_id": int(parts[0]), "task_id": int(parts[1])}
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
        # Prompt-induced dotted-number forms: "dependencies": [0.0], [0.1],
        # or a bare 0.0 when there is a single prerequisite. Distinct from
        # quoted "0.0" / "[0.0]" strings and from integer pairs [0, 0].
        if isinstance(value, float) and not isinstance(value, bool):
            return [_coerce_dotted_float_ref(value)]
        if isinstance(value, list):
            return [_coerce_dotted_float_ref(item) for item in value]
        return value

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
