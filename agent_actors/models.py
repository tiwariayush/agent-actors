from typing import List

from pydantic import BaseModel, Field, root_validator

# Keys a Plan model uses instead of child_id after reading the prompt's
# "assigned team member id" / "team members" wording. Distinct from
# worker_id, which comes from the prompt's [worker #.task #] form.
_MEMBER_ID_ALIASES = ("member_id", "team_member_id")


def _promote_member_id(values):
    """Fill missing child_id from member_id or team_member_id.

    The Plan prompt describes workers as team members and labels the JSON
    field ``<assigned team member id>``. Models often emit member_id or
    team_member_id instead of child_id on a task or dependency object.
    Without this, ParentAgent.run raises ValidationError after json.loads
    succeeds.

    Leave mappings that already have child_id or worker_id untouched so
    the existing worker_id alias repair remains independent.
    """
    if not isinstance(values, dict):
        return values
    if values.get("child_id") is not None:
        return values
    if "worker_id" in values:
        return values
    for key in _MEMBER_ID_ALIASES:
        if key not in values:
            continue
        raw = values[key]
        if isinstance(raw, bool) or raw is None:
            return values
        promoted = dict(values)
        promoted["child_id"] = promoted.pop(key)
        return promoted
    return values


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @root_validator(pre=True)
    def _alias_member_id(cls, values):
        # Without this, ParentAgent.run raises ValidationError when Plan JSON
        # uses member_id / team_member_id (from the prompt's team-member
        # wording) and omits child_id on a task or dependency object.
        return _promote_member_id(values)

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
