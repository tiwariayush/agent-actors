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
_promote_worker_id = models._promote_worker_id


class PromoteWorkerIdTests(unittest.TestCase):
    def test_worker_id_fills_missing_child_id(self):
        self.assertEqual(
            _promote_worker_id(
                {"worker_id": 0, "task_id": 0, "task": "Research AGI"}
            ),
            {"child_id": 0, "task_id": 0, "task": "Research AGI"},
        )

    def test_nonzero_worker_id(self):
        self.assertEqual(
            _promote_worker_id({"worker_id": 42, "task_id": 1}),
            {"child_id": 42, "task_id": 1},
        )

    def test_does_not_overwrite_explicit_child_id(self):
        raw = {"worker_id": 9, "child_id": 0, "task_id": 1, "task": "Keep"}
        self.assertIs(_promote_worker_id(raw), raw)

    def test_child_id_zero_is_not_treated_as_missing(self):
        raw = {"worker_id": 9, "child_id": 0, "task_id": 0}
        self.assertIs(_promote_worker_id(raw), raw)

    def test_missing_worker_id_left_unchanged(self):
        raw = {"task_id": 0, "task": "Research"}
        self.assertIs(_promote_worker_id(raw), raw)

    def test_null_worker_id_left_unchanged(self):
        raw = {"worker_id": None, "task_id": 0, "task": "Research"}
        self.assertIs(_promote_worker_id(raw), raw)

    def test_bool_worker_id_left_unchanged(self):
        raw = {"worker_id": False, "task_id": 0, "task": "Research"}
        self.assertIs(_promote_worker_id(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_promote_worker_id(0), 0)

    def test_dotted_id_field_left_unchanged(self):
        """Dotted `id` stays with draft PR #93."""
        raw = {"id": "0.0", "task": "Research"}
        self.assertIs(_promote_worker_id(raw), raw)

    def test_numeric_id_field_left_unchanged(self):
        """Numeric `id` stays with draft PR #94."""
        raw = {"id": 0.0, "task": "Research"}
        self.assertIs(_promote_worker_id(raw), raw)


class WorkerIdTaskRecordTests(unittest.TestCase):
    def test_task_with_worker_id_constructs(self):
        record = TaskRecord(
            **{
                "worker_id": 0,
                "task_id": 0,
                "task": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_worker_id(self):
        """Plan JSON `{"worker_id": 0}` is valid JSON then crashed on TaskRecord."""
        payload = json.loads(
            '{"worker_id": 0, "task_id": 0, "task": "Look up Sergey Brin\'s age",'
            ' "dependencies": []}'
        )
        record = TaskRecord(**payload)
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.id, "0.0")

    def test_dependency_object_with_worker_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"worker_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_canonical_child_id_still_works(self):
        record = TaskRecord(
            task_id=0,
            child_id=1,
            task="Research AGI",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.id, "1.0")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_explicit_child_id_wins_over_conflicting_worker_id(self):
        record = TaskRecord(
            **{
                "worker_id": 9,
                "child_id": 0,
                "task_id": 1,
                "task": "Keep assigned ids",
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 1)

    def test_parent_plan_comprehension_accepts_worker_id(self):
        plan_json = [
            {
                "worker_id": 0,
                "task_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "worker_id": 1,
                "task_id": 0,
                "task": "Write the report",
                "dependencies": [{"worker_id": 0, "task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[1].id, "1.0")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_string_worker_id_coerces(self):
        record = TaskRecord(
            **{"worker_id": "2", "task_id": 0, "task": "Write the report"}
        )
        self.assertEqual(record.child_id, 2)

    def test_dotted_id_still_rejected(self):
        """Dotted `id` stays with draft PR #93."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0.0", "task": "Research AGI"})

    def test_numeric_id_still_rejected(self):
        """Numeric `id` stays with draft PR #94."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0.0, "task": "Research AGI"})

    def test_task_id_dotted_string_still_rejected(self):
        """task_id: '0.0' with child_id set stays unhandled."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": "0.0",
                    "task": "Research AGI",
                }
            )

    def test_integer_id_without_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0, "task": "Research AGI"})

    def test_missing_child_and_worker_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"task_id": 0, "task": "Research AGI"})


class WorkerIdTaskRefTests(unittest.TestCase):
    def test_task_ref_from_worker_id(self):
        ref = TaskRef(**{"worker_id": 42, "task_id": 7})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 7)
        self.assertEqual(ref.id, "42.7")


if __name__ == "__main__":
    unittest.main()
