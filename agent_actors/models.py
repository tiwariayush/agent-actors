import json
from typing import List

from pydantic import BaseModel, Field, validator


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


def _stringified_dependency_object_list_items(value):
    """Unwrap list items that JSON-encode a single non-empty dependency object.

    Plan models sometimes JSON-encode each prerequisite as a string inside an
    otherwise-valid array, so the field is
    ``['{"child_id": 0, "task_id": 0}']`` rather than a list of objects.
    ``json.loads`` of the outer document succeeds; Pydantic then raises
    ``ValidationError: value is not a valid dict`` and ParentAgent.run aborts.

    A string field (stringified array or stringified bare object) stays with
    those repairs. Citation strings, integer pairs, and empty encodings stay
    with those repairs.
    """
    if not isinstance(value, list) or not value:
        return value
    coerced = []
    changed = False
    for item in value:
        if isinstance(item, str):
            try:
                parsed = json.loads(item.strip())
            except json.JSONDecodeError:
                coerced.append(item)
                continue
            if isinstance(parsed, dict) and parsed:
                coerced.append(parsed)
                changed = True
                continue
        coerced.append(item)
    return coerced if changed else value


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @validator("dependencies", pre=True)
    def _coerce_stringified_dependency_object_list_items(cls, value):
        # Without this, ParentAgent.run crashes when Plan JSON stringifies
        # dependency objects as list items after json.loads succeeds.
        return _stringified_dependency_object_list_items(value)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
