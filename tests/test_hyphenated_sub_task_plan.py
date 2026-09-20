import importlib.util
import json
import unittest
from pathlib import Path

from pydantic import ValidationError


def load_models_module():
    """Load models.py directly to avoid importing Ray via package __init__."""
    path = Path(__file__).resolve().parents[1] / "agent_actors" / "models.py"
    spec = importlib.util.spec_from_file_location("agent_actors_models", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


models = load_models_module()
TaskRecord = models.TaskRecord
unwrap_hyphenated_sub_task_plan = models.unwrap_hyphenated_sub_task_plan


def _task(child_id=0, task_id=0, task="Research AGI", dependencies=None):
    return {
        "child_id": child_id,
        "task_id": task_id,
        "task": task,
        "dependencies": [] if dependencies is None else dependencies,
    }


class UnwrapHyphenatedSubTaskPlanTests(unittest.TestCase):
    def test_list_plan_unchanged(self):
        tasks = [_task()]
        self.assertIs(unwrap_hyphenated_sub_task_plan(tasks), tasks)

    def test_sub_tasks_wrapper_unwrapped(self):
        """LLMs copy the prompt's hyphenated 'sub-tasks' as the array key."""
        inner = [_task()]
        self.assertIs(
            unwrap_hyphenated_sub_task_plan({"sub-tasks": inner}), inner
        )

    def test_sub_task_wrapper_unwrapped(self):
        inner = [_task(task="Write the executive report")]
        self.assertIs(
            unwrap_hyphenated_sub_task_plan({"sub-task": inner}), inner
        )

    def test_multiple_tasks_under_sub_tasks(self):
        inner = [
            _task(task="Look up Sergey Brin's age"),
            _task(
                child_id=0,
                task_id=1,
                task="Multiply the age by 12",
                dependencies=[{"child_id": 0, "task_id": 0}],
            ),
        ]
        self.assertEqual(
            unwrap_hyphenated_sub_task_plan({"sub-tasks": inner}), inner
        )

    def test_single_task_dict_under_sub_tasks_wrapped(self):
        task = _task()
        self.assertEqual(
            unwrap_hyphenated_sub_task_plan({"sub-tasks": task}), [task]
        )

    def test_single_task_dict_under_sub_task_wrapped(self):
        task = _task(task="Write the executive report")
        self.assertEqual(
            unwrap_hyphenated_sub_task_plan({"sub-task": task}), [task]
        )

    def test_prefers_sub_tasks_over_sub_task(self):
        first = [_task(task="From sub-tasks")]
        second = [_task(task="From sub-task")]
        self.assertIs(
            unwrap_hyphenated_sub_task_plan(
                {"sub-tasks": first, "sub-task": second}
            ),
            first,
        )

    def test_empty_sub_tasks_list_unwrapped(self):
        self.assertEqual(unwrap_hyphenated_sub_task_plan({"sub-tasks": []}), [])

    def test_tasks_wrapper_left_unchanged(self):
        """``{"tasks": [...]}`` stays with draft PR #74."""
        raw = {"tasks": [_task()]}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_subtasks_wrapper_left_unchanged(self):
        raw = {"subtasks": [_task()]}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_sub_tasks_underscore_wrapper_left_unchanged(self):
        raw = {"sub_tasks": [_task()]}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_plan_wrapper_left_unchanged(self):
        raw = {"plan": [_task()]}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_bare_task_object_left_unchanged(self):
        """Bare task object stays with draft PR #73."""
        raw = _task()
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_numeric_keyed_object_left_unchanged(self):
        """Numeric-keyed objects stay with draft PR #92."""
        raw = {"0": _task(), "1": _task(task_id=1, task="Write the report")}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_dotted_keyed_object_left_unchanged(self):
        """Dotted-citation-keyed objects stay with draft PR #95."""
        raw = {"0.0": _task(), "1.0": _task(child_id=1, task="Write the report")}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_task_object_with_hyphenated_text_field_left_unchanged(self):
        """Hyphenated *text* field stays a task object, not a plan wrapper."""
        raw = {
            "child_id": 0,
            "task_id": 0,
            "sub-task": "Research AGI",
            "dependencies": [],
        }
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_string_value_under_sub_tasks_left_unchanged(self):
        raw = {"sub-tasks": "Research AGI"}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(unwrap_hyphenated_sub_task_plan(0), 0)
        self.assertIsNone(unwrap_hyphenated_sub_task_plan(None))

    def test_empty_mapping_left_unchanged(self):
        raw = {}
        self.assertIs(unwrap_hyphenated_sub_task_plan(raw), raw)


class HyphenatedSubTaskPlanRecordTests(unittest.TestCase):
    def test_json_loads_sub_tasks_wrapper_then_constructs(self):
        """Plan JSON `{"sub-tasks": [...]}` is valid JSON then crashed on iterate."""
        payload = json.loads(
            '{"sub-tasks": [{"child_id": 0, "task_id": 0,'
            ' "task": "Look up Sergey Brin\'s age", "dependencies": []}]}'
        )
        self.assertIn("sub-tasks", payload)
        tasks = unwrap_hyphenated_sub_task_plan(payload)
        records = [TaskRecord(**t) for t in tasks]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].child_id, 0)
        self.assertEqual(records[0].task_id, 0)
        self.assertEqual(records[0].task, "Look up Sergey Brin's age")
        self.assertEqual(records[0].id, "0.0")

    def test_json_loads_sub_task_wrapper_then_constructs(self):
        payload = json.loads(
            '{"sub-task": [{"child_id": 2, "task_id": 0,'
            ' "task": "Write the executive report", "dependencies": []}]}'
        )
        records = [
            TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(payload)
        ]
        self.assertEqual(records[0].child_id, 2)
        self.assertEqual(records[0].task, "Write the executive report")
        self.assertEqual(records[0].id, "2.0")

    def test_parent_plan_comprehension_accepts_hyphenated_wrapper(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = {
            "sub-tasks": [
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                },
                {
                    "child_id": 1,
                    "task_id": 0,
                    "task": "Write the report",
                    "dependencies": [{"child_id": 0, "task_id": 0}],
                },
            ]
        }
        records = [
            TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(plan_json)
        ]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")
        self.assertEqual(records[1].id, "1.0")
        self.assertEqual(records[1].task, "Write the report")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_single_task_object_under_sub_tasks_constructs(self):
        plan_json = {
            "sub-tasks": {
                "child_id": 0,
                "task_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            }
        }
        records = [
            TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(plan_json)
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, "0.0")

    def test_canonical_array_still_works(self):
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            }
        ]
        records = [
            TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(plan_json)
        ]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")

    def test_iterating_unwrapped_tasks_wrapper_still_typeerrors(self):
        """``{"tasks": [...]}`` stays with draft PR #74; still TypeError on main."""
        wrapped = {"tasks": [_task()]}
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(wrapped)]

    def test_iterating_bare_task_object_still_typeerrors(self):
        """Bare task object stays with draft PR #73."""
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(_task())]

    def test_iterating_subtasks_wrapper_still_typeerrors(self):
        wrapped = {"subtasks": [_task()]}
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(wrapped)]

    def test_iterating_underscore_sub_tasks_wrapper_still_typeerrors(self):
        wrapped = {"sub_tasks": [_task()]}
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(wrapped)]

    def test_hyphenated_text_field_still_rejected(self):
        """``sub-task`` as task *text* is not filled; #110 aliases sub_task."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub-task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_sub_task_underscore_text_still_rejected(self):
        """``sub_task`` text stays with draft PR #110."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub_task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_iterating_numeric_keyed_object_still_typeerrors(self):
        """Numeric-keyed objects stay with draft PR #92."""
        wrapped = {"0": _task()}
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(wrapped)]

    def test_iterating_dotted_keyed_object_still_typeerrors(self):
        """Dotted-citation-keyed objects stay with draft PR #95."""
        wrapped = {"0.0": _task()}
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in unwrap_hyphenated_sub_task_plan(wrapped)]


if __name__ == "__main__":
    unittest.main()
