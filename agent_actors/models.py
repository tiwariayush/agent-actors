import re
from typing import List

from pydantic import BaseModel, Field, validator

# The Plan prompt says: "When referencing other tasks, use the format
# [worker #.task #]." Models that follow that sentence emit the citation
# literally ("[worker 0.task 0]") instead of a {child_id, task_id} object.
# Dotted "[0.0]" refs, "[0, 0]" pairs, and comma-joined "0.0, 1.0" strings
# are different shapes and stay unparsed here.
_WORKER_TASK_CITE = re.compile(
    r"\[\s*worker\s+#?\s*(?P<child>\d+)\s*\.\s*task\s+#?\s*(?P<task>\d+)\s*\]",
    re.IGNORECASE,
)


def _parse_worker_task_citations(value):
    """Return TaskRef mappings if *value* is only [worker #.task #] citations."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    matches = list(_WORKER_TASK_CITE.finditer(stripped))
    if not matches:
        return None
    remainder = _WORKER_TASK_CITE.sub("", stripped)
    if re.sub(r"[\s,]+", "", remainder):
        return None
    return [
        {"child_id": int(match.group("child")), "task_id": int(match.group("task"))}
        for match in matches
    ]


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
    def _normalize_worker_task_citations(cls, value):
        # Prompt-literal citations: "[worker 0.task 0]" or several joined by
        # commas. json.loads accepts them, then TaskRecord(**t) raises
        # ValidationError and ParentAgent.run aborts before any child work.
        refs = _parse_worker_task_citations(value)
        if refs is not None:
            return refs
        if isinstance(value, list):
            coerced = []
            expanded = False
            for item in value:
                item_refs = _parse_worker_task_citations(item)
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
