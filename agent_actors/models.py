import re
from typing import List

from pydantic import BaseModel, Field, validator

# Plan prompt tells models to cite other tasks as [worker #.task #]. For a
# task with several prerequisites, models often concatenate those citations
# into one string ("0.0, 1.0" or "[0.0], [1.0]") instead of an object array.
_DOTTED_TASK_REF = re.compile(r"\[(\d+)\.(\d+)\]|(\d+)\.(\d+)")


def _parse_comma_joined_task_refs(value):
    """Return TaskRef mappings if *value* is two or more dotted task refs.

    Single refs (``"0.0"`` / ``"[0.0]"``) stay with draft PR #76.
    Integer pairs (``"[0, 0]"``) stay with draft PR #77.
    Stringified empty lists stay with draft PR #90.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    matches = list(_DOTTED_TASK_REF.finditer(stripped))
    if len(matches) < 2:
        return None
    remainder = _DOTTED_TASK_REF.sub("", stripped)
    if re.sub(r"[\s,\[\]]+", "", remainder):
        return None
    refs = []
    for match in matches:
        child_id = match.group(1) or match.group(3)
        task_id = match.group(2) or match.group(4)
        refs.append({"child_id": int(child_id), "task_id": int(task_id)})
    return refs


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
        # Multi-ref citation strings: "0.0, 1.0", "[0.0], [1.0]", "[0.0, 1.0]".
        # Without coercion, ParentAgent.run crashes in TaskRecord(**t).
        refs = _parse_comma_joined_task_refs(value)
        if refs is not None:
            return refs
        if isinstance(value, list):
            coerced = []
            expanded = False
            for item in value:
                item_refs = _parse_comma_joined_task_refs(item)
                if item_refs is not None:
                    coerced.extend(item_refs)
                    expanded = True
                else:
                    coerced.append(item)
            if expanded:
                return coerced
        return value

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
