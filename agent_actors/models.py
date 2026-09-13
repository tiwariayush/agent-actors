import json
from typing import List

from pydantic import BaseModel, Field, validator


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


def _stringified_dependency_object(value):
    """Unwrap a string that JSON-encodes a single non-empty dependency object.

    Plan models sometimes JSON-encode one prerequisite as a string, so the
    field is ``'{"child_id": 0, "task_id": 0}'`` rather than a JSON array or
    a parsed object. ``json.loads`` of the outer document succeeds; Pydantic
    then raises ``ValidationError: value is not a valid list`` and
    ParentAgent.run aborts.

    Empty encodings (``"{}"`` / ``"[]"`` / ``""`` / ``"null"``) stay with the
    empty-string repair. A parsed dict stays with the bare-object wrap. A
    stringified object array stays with that repair.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value.strip())
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict) and parsed:
        return [parsed]
    return value


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @validator("dependencies", pre=True)
    def _coerce_stringified_dependency_object(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON stringifies a
        # single dependency object after json.loads succeeds.
        return _stringified_dependency_object(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
