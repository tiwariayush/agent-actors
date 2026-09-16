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
_default_empty_list_task = models._default_empty_list_task


class DefaultEmptyListTaskTests(unittest.TestCase):
    def test_empty_list_becomes_empty_string(self):
        self.assertEqual(_default_empty_list_task([]), "")

    def test_plain_string_left_unchanged(self):
        raw = "Research AGI"
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_empty_string_left_unchanged(self):
        raw = ""
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_null_task_left_unchanged(self):
        """Null task stays with the existing null-task repair."""
        self.assertIsNone(_default_empty_list_task(None))

    def test_non_empty_string_list_left_unchanged(self):
        """Non-empty list-valued task stays with the existing list-task repair."""
        raw = ["Research AGI", "Write the report"]
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_single_element_list_left_unchanged(self):
        raw = ["Look up Sergey Brin's age"]
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_list_of_empty_string_left_unchanged(self):
        raw = [""]
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_mixed_type_list_left_unchanged(self):
        raw = ["Research AGI", {"step": "Write"}]
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_list_of_null_left_unchanged(self):
        raw = [None]
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_list_of_objects_left_unchanged(self):
        raw = [{"description": "Research AGI"}]
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_empty_object_left_unchanged(self):
        """Empty object task stays with the existing empty-object repair."""
        raw = {}
        self.assertIs(_default_empty_list_task(raw), raw)

    def test_object_task_left_unchanged(self):
        raw = {"description": "Research AGI"}
        self.assertIs(_default_empty_list_task(raw), raw)


class EmptyListTaskRecordTests(unittest.TestCase):
    def test_empty_list_constructs(self):
        record = TaskRecord(
            child_id=0,
            task_id=0,
            task=[],
            dependencies=[],
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_empty_list_task(self):
        """Plan JSON `"task": []` is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0, "task": [], "dependencies": []}'
        )
        self.assertEqual(payload["task"], [])
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_canonical_string_task_still_works(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task, "Synthesize")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_parent_plan_comprehension_accepts_empty_list_task(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": [],
                "dependencies": [],
            },
            {
                "child_id": 2,
                "task_id": 0,
                "task": "Write the executive report",
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].task, "Write the executive report")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_null_task_still_rejected(self):
        """JSON-null task remains a separate null-task repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": None,
                    "dependencies": [],
                }
            )

    def test_missing_task_still_rejected(self):
        """Omitted task key remains a separate repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "dependencies": [],
                }
            )

    def test_non_empty_list_task_still_rejected(self):
        """Non-empty list-valued task remains draft PR #98's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": ["Research AGI", "Write the report"],
                    "dependencies": [],
                }
            )

    def test_single_element_list_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": ["Research AGI"],
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

    def test_list_of_null_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": [None],
                    "dependencies": [],
                }
            )

    def test_empty_object_still_rejected(self):
        """Empty object task remains draft PR #107's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {},
                    "dependencies": [],
                }
            )

    def test_description_object_still_rejected(self):
        """Known-key object tasks remain draft PR #102's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"description": "Research AGI"},
                    "dependencies": [],
                }
            )

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "task_id": 0,
                    "task": [],
                    "dependencies": [],
                }
            )


class EmptyListTaskRefTests(unittest.TestCase):
    def test_task_ref_does_not_require_task_field(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_dependency_object_ignores_empty_list_task(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Write the report",
            dependencies=[
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": [],
                }
            ],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.task, "Write the report")


if __name__ == "__main__":
    unittest.main()
