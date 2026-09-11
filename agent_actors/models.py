from typing import List

from pydantic import BaseModel, Field, root_validator

# Keys a Plan model uses when it puts the objective next to child_id/task_id
# instead of under the required `task` field.
_TASK_SIBLING_STRING_KEYS = (
    "description",
    "text",
    "objective",
    "instruction",
)


def _fill_missing_task(values):
    """Fill a missing TaskRecord.task from sibling strings, else "".

    Plan models sometimes omit the task key while still setting child_id and
    task_id, or put the objective in description/text/objective/instruction.
    That is valid JSON, but TaskRecord requires task, so ParentAgent.run
    raises ValidationError before any child work starts.

    JSON-null, list-valued, and object-shaped task fields stay untouched so
    those draft repairs remain independent.
    """
    if not isinstance(values, dict) or "task" in values:
        return values
    filled = dict(values)
    for key in _TASK_SIBLING_STRING_KEYS:
        item = values.get(key)
        if isinstance(item, str) and item:
            filled["task"] = item
            return filled
    filled["task"] = ""
    return filled


class TaskRef(BaseModel):
    child_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.child_id}.{self.task_id}"


class TaskRecord(TaskRef):
    task: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    @root_validator(pre=True)
    def _coerce_missing_task(cls, values):
        # Without this, ParentAgent.run crashes when Plan JSON omits the
        # task key after json.loads succeeds.
        return _fill_missing_task(values)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.task}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"
