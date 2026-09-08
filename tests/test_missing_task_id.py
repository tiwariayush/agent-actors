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
_default_missing_task_id = models._default_missing_task_id


class DefaultMissingTaskIdTests(unittest.TestCase):
    def test_missing_task_id_defaults_to_zero_when_child_id_present(self):
        self.assertEqual(
            _default_missing_task_id(
                {"child_id": 0, "task": "Research AGI", "dependencies": []}
            ),
            {
                "child_id": 0,
                "task": "Research AGI",
                "dependencies": [],
                "task_id": 0,
            },
        )

    def test_null_task_id_defaults_to_zero_when_child_id_present(self):
        self.assertEqual(
            _default_missing_task_id(
                {
                    "child_id": 2,
                    "task_id": None,
                    "task": "Write the report",
                }
            ),
            {
                "child_id": 2,
                "task_id": 0,
                "task": "Write the report",
            },
        )

    def test_nonzero_child_id_still_defaults_task_id_to_zero(self):
        self.assertEqual(
            _default_missing_task_id({"child_id": 42, "task": "Draft jokes"}),
            {"child_id": 42, "task": "Draft jokes", "task_id": 0},
        )

    def test_does_not_overwrite_explicit_task_id(self):
        raw = {"child_id": 0, "task_id": 1, "task": "Keep"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_explicit_zero_task_id_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 0, "task": "Keep"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_missing_child_id_left_unchanged(self):
        raw = {"task": "Research AGI"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_null_child_id_left_unchanged(self):
        """Null child_id cannot identify a worker."""
        raw = {"child_id": None, "task": "Research AGI"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_quoted_digit_id_left_unchanged(self):
        """Quoted digit id stays with the existing digit-string-id repair."""
        raw = {"id": "0", "child_id": 0, "task": "Research AGI"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_integer_id_left_unchanged(self):
        """Unquoted integer id stays with the existing numeric-id repair."""
        raw = {"id": 0, "child_id": 0, "task": "Research AGI"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_dotted_string_id_left_unchanged(self):
        """Dotted citations stay with the existing combined-id repair."""
        raw = {"id": "0.0", "task": "Research AGI"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_float_id_left_unchanged(self):
        raw = {"id": 0.0, "child_id": 0, "task": "Research AGI"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_worker_id_without_child_id_left_unchanged(self):
        """worker_id aliasing stays with the existing worker_id repair."""
        raw = {"worker_id": 0, "task": "Research AGI"}
        self.assertIs(_default_missing_task_id(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_default_missing_task_id("0"), "0")


class MissingTaskIdRecordTests(unittest.TestCase):
    def test_task_without_task_id_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_omitted_task_id(self):
        """Plan JSON that omits task_id is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task": "Research AGI", "dependencies": []}'
        )
        self.assertNotIn("task_id", payload)
        record = TaskRecord(**payload)
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.id, "0.0")

    def test_json_loads_null_task_id(self):
        payload = json.loads(
            '{"child_id": 2, "task_id": null,'
            ' "task": "Write the executive report", "dependencies": []}'
        )
        self.assertIsNone(payload["task_id"])
        record = TaskRecord(**payload)
        self.assertEqual(record.child_id, 2)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.id, "2.0")

    def test_dependency_object_missing_task_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Multiply that age by 12",
            dependencies=[{"child_id": 0}],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_canonical_fields_still_work(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.id, "0.1")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_explicit_task_id_wins(self):
        record = TaskRecord(
            child_id=0,
            task_id=3,
            task="Keep assigned ids",
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 3)
        self.assertEqual(record.id, "0.3")

    def test_parent_plan_comprehension_accepts_omitted_task_id(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "child_id": 2,
                "task": "Write the executive report",
                "dependencies": [{"child_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].dependencies[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")
        self.assertEqual(records[1].task, "Write the executive report")

    def test_quoted_digit_id_still_rejected(self):
        """\"id\": \"0\" remains a separate digit-string-id repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0", "child_id": 0, "task": "Research AGI"})

    def test_integer_id_with_child_id_still_rejected(self):
        """Unquoted integer id remains a separate numeric-id repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0, "child_id": 0, "task": "Research AGI"})

    def test_dotted_string_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0.0", "task": "Research AGI"})

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"task": "Research AGI", "dependencies": []})

    def test_null_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": None,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

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

    def test_integer_id_without_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0, "task": "Research AGI"})


class MissingTaskIdRefTests(unittest.TestCase):
    def test_task_ref_from_child_id_only(self):
        ref = TaskRef(**{"child_id": 42})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 0)
        self.assertEqual(ref.id, "42.0")

    def test_task_ref_null_task_id(self):
        ref = TaskRef(**{"child_id": 1, "task_id": None})
        self.assertEqual(ref.child_id, 1)
        self.assertEqual(ref.task_id, 0)


if __name__ == "__main__":
    unittest.main()
