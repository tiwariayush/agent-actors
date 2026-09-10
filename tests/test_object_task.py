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
_task_from_object = models._task_from_object


class TaskFromObjectTests(unittest.TestCase):
    def test_description_key(self):
        self.assertEqual(
            _task_from_object({"description": "Research AGI"}),
            "Research AGI",
        )

    def test_nested_task_key(self):
        self.assertEqual(
            _task_from_object({"task": "Look up Sergey Brin's age"}),
            "Look up Sergey Brin's age",
        )

    def test_text_key(self):
        self.assertEqual(
            _task_from_object({"text": "Write the executive report"}),
            "Write the executive report",
        )

    def test_objective_key(self):
        self.assertEqual(
            _task_from_object({"objective": "Synthesize findings"}),
            "Synthesize findings",
        )

    def test_instruction_key(self):
        self.assertEqual(
            _task_from_object({"instruction": "Include 5 jokes and quotes"}),
            "Include 5 jokes and quotes",
        )

    def test_prefers_task_over_description(self):
        self.assertEqual(
            _task_from_object(
                {"task": "Research AGI", "description": "ignored extra"}
            ),
            "Research AGI",
        )

    def test_plain_string_left_unchanged(self):
        raw = "Research AGI"
        self.assertIs(_task_from_object(raw), raw)

    def test_null_task_left_unchanged(self):
        """Null task stays with the existing null-task repair."""
        self.assertIsNone(_task_from_object(None))

    def test_list_task_left_unchanged(self):
        """List-valued task stays with the existing list-task repair."""
        raw = ["Research AGI", "Write the report"]
        self.assertIs(_task_from_object(raw), raw)

    def test_empty_list_left_unchanged(self):
        raw = []
        self.assertIs(_task_from_object(raw), raw)

    def test_empty_object_left_unchanged(self):
        raw = {}
        self.assertIs(_task_from_object(raw), raw)

    def test_unknown_keys_left_unchanged(self):
        raw = {"notes": "Research AGI"}
        self.assertIs(_task_from_object(raw), raw)

    def test_nested_non_string_description_left_unchanged(self):
        raw = {"description": {"text": "Research AGI"}}
        self.assertIs(_task_from_object(raw), raw)


class ObjectTaskRecordTests(unittest.TestCase):
    def test_description_object_constructs(self):
        record = TaskRecord(
            child_id=0,
            task_id=0,
            task={"description": "Research AGI"},
            dependencies=[],
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Research AGI")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_task_object(self):
        """Plan JSON `\"task\": {\"description\": \"...\"}` is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0,'
            ' "task": {"description": "Look up Sergey Brin\'s age"},'
            ' "dependencies": []}'
        )
        self.assertIsInstance(payload["task"], dict)
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_nested_task_key_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 2,
                "task_id": 0,
                "task": {"task": "Write the executive report"},
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 2)
        self.assertEqual(record.task, "Write the executive report")
        self.assertEqual(record.id, "2.0")

    def test_canonical_string_task_still_works(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task, "Synthesize")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_parent_plan_comprehension_accepts_object_task(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": {"description": "Research AGI"},
                "dependencies": [],
            },
            {
                "child_id": 2,
                "task_id": 0,
                "task": {"task": "Write the executive report"},
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].task, "Write the executive report")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_nested_task_record_shaped_object(self):
        """Models sometimes nest a whole task object inside the task field."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0,'
            ' "task": {"child_id": 0, "task_id": 0,'
            ' "task": "Research AGI", "dependencies": []},'
            ' "dependencies": []}'
        )
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "Research AGI")
        self.assertEqual(record.id, "0.0")

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

    def test_list_task_still_rejected(self):
        """List-valued task remains a separate list-task repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": ["Research AGI", "Write the report"],
                    "dependencies": [],
                }
            )

    def test_empty_object_task_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {},
                    "dependencies": [],
                }
            )

    def test_unknown_key_object_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"notes": "Research AGI"},
                    "dependencies": [],
                }
            )

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "task_id": 0,
                    "task": {"description": "Research AGI"},
                    "dependencies": [],
                }
            )


class ObjectTaskRefTests(unittest.TestCase):
    def test_task_ref_does_not_require_task_field(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_dependency_object_ignores_object_task(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Write the report",
            dependencies=[
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"description": "Research AGI"},
                }
            ],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.task, "Write the report")


if __name__ == "__main__":
    unittest.main()
