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


def _is_null_plan_task_placeholder(item) -> bool:
    """True for null entries or objects with all primary task fields unset."""
    if item is None:
        return True
    if isinstance(item, dict):
        return (
            item.get("task_id") is None
            and item.get("child_id") is None
            and item.get("task") is None
        )
    return False


def drop_null_plan_task_placeholders(raw_plan):
    """Drop null Plan task placeholders so ParentAgent.run does not crash.

    The Plan prompt always shows a JSON array of task objects. When the model
    decides no (or fewer) sub-tasks are needed, it often keeps that array shape
    but nulls the placeholder (``[null]``, ``[{}]``, or objects with all primary
    fields unset), or returns JSON ``null``. Iterating those values raises
    ``TypeError`` while building ``TaskRecord``s and aborts before any child
    work starts.

    Distinct from draft PR #79 (null *dependency* placeholders inside a task)
    and PR #71 (top-level ``"dependencies": null``).
    """
    if raw_plan is None:
        return []
    if not isinstance(raw_plan, list):
        # Leave single-object / wrapper shapes to draft PRs #73 / #74.
        return raw_plan
    return [
        item for item in raw_plan if not _is_null_plan_task_placeholder(item)
    ]


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
                for t in drop_null_plan_task_placeholders(
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
