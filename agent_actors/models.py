import json
from typing import List

from pydantic import BaseModel, Field, validator


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


def _empty_stringified_dependencies(value):
    """Return [] if *value* is a string encoding no dependencies.

    Plan models often JSON-encode an empty dependency list, so the field is
    the string ``"[]"`` (or ``""`` / ``"null"``) rather than a JSON array.
    ``json.loads`` of the outer document succeeds; Pydantic then raises
    ``ValidationError: value is not a valid list`` and ParentAgent.run aborts.
    Non-empty string forms such as ``"0.0"`` / ``"[0.0]"`` are left unchanged.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped == "":
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if parsed in (None, [], {}):
        return []
    return None


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @validator("dependencies", pre=True)
    def _normalize_dependencies(cls, value):
        empty = _empty_stringified_dependencies(value)
        if empty is not None:
            return empty
        return value

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
