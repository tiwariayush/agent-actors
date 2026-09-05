import re
from typing import List

from pydantic import BaseModel, Field, root_validator

# Plan prompt identifies tasks as [worker #.task #]; models often put that
# citation into child_id or task_id instead of using plain integers.
_CITATION_ID = re.compile(r"^\[?(?P<child>\d+)\.(?P<task>\d+)\]?$")


def _citation_match(value):
    if not isinstance(value, str):
        return None
    return _CITATION_ID.fullmatch(value.strip())


def _promote_dotted_field_ids(values):
    """Coerce citation-shaped child_id/task_id ('0.0' / '[0.0]') to ints.

    When the other id is already present, only that field's half is taken
    from the citation so an explicit child_id/task_id is not overwritten.
    A missing half is filled from the citation. The `id` field is left
    untouched so draft PRs #93/#94 stay independent.
    """
    if not isinstance(values, dict):
        return values

    child_cite = _citation_match(values.get("child_id"))
    task_cite = _citation_match(values.get("task_id"))
    if child_cite is None and task_cite is None:
        return values

    promoted = dict(values)
    if child_cite is not None:
        promoted["child_id"] = int(child_cite.group("child"))
        if task_cite is None and "task_id" not in values:
            promoted["task_id"] = int(child_cite.group("task"))
    if task_cite is not None:
        promoted["task_id"] = int(task_cite.group("task"))
        if child_cite is None and "child_id" not in values:
            promoted["child_id"] = int(task_cite.group("child"))
    return promoted


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _split_dotted_field_ids(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses the prompt's [worker #.task #] form as child_id or task_id.
        return _promote_dotted_field_ids(values)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
