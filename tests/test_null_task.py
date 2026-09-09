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
_default_null_task = models._default_null_task


class DefaultNullTaskTests(unittest.TestCase):
    def test_null_becomes_empty_string(self):
        self.assertEqual(_default_null_task(None), "")

    def test_plain_string_left_unchanged(self):
        raw = "Research AGI"
        self.assertIs(_default_null_task(raw), raw)

    def test_empty_string_left_unchanged(self):
        raw = ""
        self.assertIs(_default_null_task(raw), raw)

    def test_list_task_left_unchanged(self):
        """List-valued task stays with the existing list-task repair."""
        raw = ["Research AGI", "Write the report"]
        self.assertIs(_default_null_task(raw), raw)

    def test_empty_list_left_unchanged(self):
        raw = []
        self.assertIs(_default_null_task(raw), raw)

    def test_object_task_left_unchanged(self):
        """Object-shaped task stays with the deferred object-task repair."""
        raw = {"description": "Research AGI"}
        self.assertIs(_default_null_task(raw), raw)

    def test_false_left_unchanged(self):
        self.assertIs(_default_null_task(False), False)


class NullTaskRecordTests(unittest.TestCase):
    def test_null_task_constructs_with_ids_set(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task_id": 0,
                "task": None,
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_null_task(self):
        """Plan JSON `\"task\": null` is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0, "task": null, "dependencies": []}'
        )
        self.assertIsNone(payload["task"])
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_nonzero_child_still_defaults_null_task(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task=None,
            dependencies=[],
        )
        self.assertEqual(record.child_id, 2)
        self.assertEqual(record.task, "")
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

    def test_explicit_empty_string_task_still_works(self):
        record = TaskRecord(
            child_id=0,
            task_id=0,
            task="",
            dependencies=[],
        )
        self.assertEqual(record.task, "")

    def test_parent_plan_comprehension_accepts_null_task(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": None,
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

    def test_prompt_placeholder_copy_as_json_null(self):
        """Models copy ``<task task>`` as JSON null while filling ids."""
        payload = json.loads(
            '[{"child_id": 0, "task_id": 0, "task": null, "dependencies": []},'
            ' {"child_id": 2, "task_id": 0, "task": null,'
            ' "dependencies": [{"child_id": 0, "task_id": 0}]}]'
        )
        records = [TaskRecord(**t) for t in payload]
        self.assertEqual([r.id for r in records], ["0.0", "2.0"])
        self.assertEqual([r.task for r in records], ["", ""])
        self.assertEqual(records[1].dependencies[0].id, "0.0")

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

    def test_object_task_still_rejected(self):
        """Object-shaped task remains a separate repair."""
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
            TaskRecord(**{"task_id": 0, "task": None, "dependencies": []})

    def test_null_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": None,
                    "task_id": 0,
                    "task": None,
                    "dependencies": [],
                }
            )


class NullTaskRefTests(unittest.TestCase):
    def test_task_ref_does_not_require_task_field(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_dependency_object_ignores_null_task(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Write the report",
            dependencies=[{"child_id": 0, "task_id": 0, "task": None}],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.task, "Write the report")


if __name__ == "__main__":
    unittest.main()
