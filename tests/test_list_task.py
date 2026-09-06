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
TaskRef = models.TaskRef
_join_list_task = models._join_list_task


class JoinListTaskTests(unittest.TestCase):
    def test_joins_two_step_strings(self):
        self.assertEqual(
            _join_list_task(
                [
                    "Write an executive report on AGI",
                    "Include 5 relevant jokes and quotes",
                ]
            ),
            "Write an executive report on AGI\nInclude 5 relevant jokes and quotes",
        )

    def test_single_element_list_unwraps(self):
        self.assertEqual(
            _join_list_task(["Look up Sergey Brin's age"]),
            "Look up Sergey Brin's age",
        )

    def test_strips_step_whitespace(self):
        self.assertEqual(
            _join_list_task(["  Research AGI  ", "Write the report"]),
            "Research AGI\nWrite the report",
        )

    def test_plain_string_left_unchanged(self):
        raw = "Research AGI"
        self.assertIs(_join_list_task(raw), raw)

    def test_null_task_left_unchanged(self):
        """Null task stays with the deferred \"task\": null repair."""
        self.assertIsNone(_join_list_task(None))

    def test_empty_list_left_unchanged(self):
        raw = []
        self.assertIs(_join_list_task(raw), raw)

    def test_non_string_list_left_unchanged(self):
        raw = ["Research AGI", {"step": "Write"}]
        self.assertIs(_join_list_task(raw), raw)

    def test_dict_task_left_unchanged(self):
        raw = {"description": "Research AGI"}
        self.assertIs(_join_list_task(raw), raw)


class ListTaskRecordTests(unittest.TestCase):
    def test_task_list_constructs(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task=[
                "Write an executive report on Artificial General Intelligence",
                "Include a list of 5 relevant jokes and quotes",
            ],
            dependencies=[],
        )
        self.assertEqual(record.child_id, 2)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(
            record.task,
            "Write an executive report on Artificial General Intelligence\n"
            "Include a list of 5 relevant jokes and quotes",
        )
        self.assertEqual(record.id, "2.0")

    def test_json_loads_task_array(self):
        """Plan JSON `\"task\": [\"step\", \"step\"]` is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0,'
            ' "task": ["Look up Sergey Brin\'s age", "Multiply that age by 12"],'
            ' "dependencies": []}'
        )
        self.assertIsInstance(payload["task"], list)
        record = TaskRecord(**payload)
        self.assertEqual(
            record.task,
            "Look up Sergey Brin's age\nMultiply that age by 12",
        )
        self.assertEqual(record.id, "0.0")

    def test_single_step_list_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task_id": 0,
                "task": ["Research AGI"],
                "dependencies": [],
            }
        )
        self.assertEqual(record.task, "Research AGI")

    def test_canonical_string_task_still_works(self):
        record = TaskRecord(
            task_id=0,
            child_id=1,
            task="Research AGI",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task, "Research AGI")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_parent_plan_comprehension_accepts_list_task(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": ["Research AGI", "Identify key papers"],
                "dependencies": [],
            },
            {
                "child_id": 2,
                "task_id": 0,
                "task": [
                    "Write the executive report",
                    "Include 5 relevant jokes and quotes",
                ],
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].task, "Research AGI\nIdentify key papers")
        self.assertEqual(
            records[1].task,
            "Write the executive report\nInclude 5 relevant jokes and quotes",
        )
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_null_task_still_rejected(self):
        """\"task\": null with ids set remains a separate repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": None,
                    "dependencies": [],
                }
            )

    def test_empty_list_task_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": [],
                    "dependencies": [],
                }
            )

    def test_mixed_type_list_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": ["Research AGI", {"step": "Write"}],
                    "dependencies": [],
                }
            )

    def test_object_task_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"description": "Research AGI"},
                    "dependencies": [],
                }
            )


class ListTaskRefTests(unittest.TestCase):
    def test_task_ref_does_not_require_task_field(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")


if __name__ == "__main__":
    unittest.main()
