import json
from typing import List

from pydantic import BaseModel, Field, validator


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


def _stringified_dependency_object_list(value):
    """Unwrap a string that JSON-encodes a non-empty dependency object list.

    Plan models sometimes JSON-encode a real dependency array, so the field is
    the string ``'[{"child_id": 0, "task_id": 0}]'`` rather than a JSON array.
    ``json.loads`` of the outer document succeeds; Pydantic then raises
    ``ValidationError: value is not a valid list`` and ParentAgent.run aborts.

    Empty encodings (``"[]"`` / ``""`` / ``"null"`` / ``"{}"``) stay with the
    empty-string repair. Citation strings (``"0.0"`` / ``"[0.0]"``) and
    integer-pair lists stay with those repairs.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value.strip())
    except json.JSONDecodeError:
        return value
    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(item, dict) for item in parsed)
    ):
        return parsed
    return value


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @validator("dependencies", pre=True)
    def _coerce_stringified_dependency_objects(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON stringifies a
        # non-empty dependency array after json.loads succeeds.
        return _stringified_dependency_object_list(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
