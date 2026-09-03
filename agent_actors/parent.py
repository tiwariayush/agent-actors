import re
from collections import defaultdict
from pprint import pprint
from typing import List

import ray
from langchain.schema import AgentAction, AgentFinish
from pydantic import Field

from agent_actors.agent import Agent
from agent_actors.chains.parent import Adjust, Plan
from agent_actors.child import ChildAgent
from agent_actors.models import TaskRecord

# Fields that distinguish a Plan task object from an unrelated mapping.
_TASK_OBJECT_KEYS = frozenset({"task", "task_id", "child_id", "id", "dependencies"})
# Plan prompt citations look like 0.0 or [0.0]; JSON object keys are strings.
_DOTTED_CITATION_KEY = re.compile(r"^\[(\d+)\.(\d+)\]$|^(\d+)\.(\d+)$")


def _parse_dotted_citation_key(key):
    if not isinstance(key, str):
        return None
    match = _DOTTED_CITATION_KEY.fullmatch(key)
    if not match:
        return None
    if match.group(1) is not None:
        return int(match.group(1)), int(match.group(2))
    return int(match.group(3)), int(match.group(4))


def _looks_like_task(value):
    return isinstance(value, dict) and bool(_TASK_OBJECT_KEYS.intersection(value))


def _tasks_from_dotted_keyed_plan(raw_plan):
    """Return task dicts if *raw_plan* is keyed by [worker #.task #] citations.

    The Plan prompt asks for a JSON array and identifies work as
    ``[worker #.task #]``. Models often emit an object keyed by that citation
    instead, e.g. ``{"0.0": {task}, "1.0": {task}}``. Iterating that object
    yields the string keys, and ``TaskRecord(**key)`` raises ``TypeError``,
    aborting the parent run before any child work starts.

    All-digit keys (``{"0": {task}}``) and wrappers such as ``{"tasks": [...]}``
    or a bare task object are left unchanged.
    """
    if not isinstance(raw_plan, dict) or not raw_plan:
        return None
    tasks = []
    for key, value in raw_plan.items():
        parsed = _parse_dotted_citation_key(key)
        if parsed is None or not _looks_like_task(value):
            return None
        child_id, task_id = parsed
        task = dict(value)
        if task.get("child_id") is None:
            task["child_id"] = child_id
        if task.get("task_id") is None:
            task["task_id"] = task_id
        tasks.append(task)
    return tasks


def normalize_plan_tasks(raw_plan):
    """Normalize dotted-citation-keyed Plan JSON into a list of task dicts."""
    if isinstance(raw_plan, list):
        return raw_plan
    dotted = _tasks_from_dotted_keyed_plan(raw_plan)
    if dotted is not None:
        return dotted
    return raw_plan


class ParentAgent(Agent):
    plan: Plan = Field(init=False)
    adjust: Adjust = Field(init=False)

    def __init__(self, *args, **kwargs):
        chain_params = dict(
            llm=kwargs["llm"],
            verbose=kwargs.get("verbose", True),
            callback_manager=kwargs.get("callback_manager", None),
        )

        super().__init__(
            *args,
            **kwargs,
            plan=Plan.from_llm(**chain_params),
            adjust=Adjust.from_llm(**chain_params),
        )

    def run(self, task: str, working_memory: List[ray.ObjectRef] = []):
        try:
            self.status = "running"
            self.task = task

            for x in working_memory:
                if isinstance(x, AgentFinish):
                    import ipdb

                    ipdb.set_trace()
            context = self.get_context() + "\n".join(ray.get(working_memory))

            planned_tasks = [
                TaskRecord(**t)
                for t in normalize_plan_tasks(
                    self.plan(
                        inputs=dict(
                            context=context,
                            task=self.task,
                            child_summary="\n\n".join(
                                f"ID: {id}\n{child.get_context()}"
                                for id, child in self.children.items()
                            ),
                        ),
                    )["json"]
                )
            ]

            if self.verbose:
                child_tasks = defaultdict(list)
                for sub_task in planned_tasks:
                    child_tasks[sub_task.child_id].append(sub_task)
                for child_id, child_tasks in child_tasks.items():
                    print(f"\n=== CHILD {child_id} TASKS ===")
                    pprint(child_tasks)

            task_result_refs = {}

            for sub_task in planned_tasks:
                if self.verbose:
                    print(f"\n\n\n=== CHILD TASK {sub_task} ===")

                child_id = sub_task.child_id

                if child_id not in self.children:
                    self.add_child(
                        ChildAgent(
                            llm=self.llm,
                            verbose=self.verbose,
                            name=f"Team Member {sub_task.child_id}",
                            traits=["focused", "team player"],
                            max_iterations=3,
                            callback_manager=self.callback_manager,
                        )
                    )

                task_result_refs[sub_task.id] = self.children[
                    child_id
                ].actor.run.remote(
                    task=sub_task.task,
                    working_memory=[
                        task_result_refs[d.id] for d in sub_task.dependencies
                    ],
                )

            task_results = []
            tasks_in_progress = list(task_result_refs.values())
            while any(tasks_in_progress):
                tasks_completed, tasks_in_progress = ray.wait(
                    tasks_in_progress,
                    num_returns=min(self.reflect_every, len(tasks_in_progress)),
                )

                results = ray.get(tasks_completed)

                for result in results:
                    task_results.append(result)

            self.pause_to_reflect()

            adjustment = self.adjust.run(
                context=self.get_context(),
                task=self.task,
                results="\n".join(task_results),
            )

            if adjustment["confidence"] >= 8:
                return AgentFinish(adjustment, "success")

            return AgentAction("Human", "What should my next task be?", "info")
        finally:
            self.task = ""
            self.status = "idle"
