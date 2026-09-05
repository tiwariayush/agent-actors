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
_promote_dotted_field_ids = models._promote_dotted_field_ids


class PromoteDottedFieldIdsTests(unittest.TestCase):
    def test_task_id_citation_keeps_explicit_child_id(self):
        self.assertEqual(
            _promote_dotted_field_ids(
                {"child_id": 0, "task_id": "0.0", "task": "Research AGI"}
            ),
            {"child_id": 0, "task_id": 0, "task": "Research AGI"},
        )

    def test_bracketed_task_id_matches_prompt_citation(self):
        self.assertEqual(
            _promote_dotted_field_ids({"child_id": 0, "task_id": "[0.0]"}),
            {"child_id": 0, "task_id": 0},
        )

    def test_does_not_overwrite_conflicting_child_id(self):
        self.assertEqual(
            _promote_dotted_field_ids(
                {"child_id": 1, "task_id": "0.0", "task": "Keep child"}
            ),
            {"child_id": 1, "task_id": 0, "task": "Keep child"},
        )

    def test_child_id_citation_keeps_explicit_task_id(self):
        self.assertEqual(
            _promote_dotted_field_ids({"child_id": "1.2", "task_id": 5}),
            {"child_id": 1, "task_id": 5},
        )

    def test_task_id_citation_fills_missing_child_id(self):
        self.assertEqual(
            _promote_dotted_field_ids({"task_id": "2.3", "task": "Research"}),
            {"task_id": 3, "task": "Research", "child_id": 2},
        )

    def test_child_id_citation_fills_missing_task_id(self):
        self.assertEqual(
            _promote_dotted_field_ids({"child_id": "[4.5]", "task": "Write"}),
            {"child_id": 4, "task": "Write", "task_id": 5},
        )

    def test_both_citation_fields_use_their_own_half(self):
        self.assertEqual(
            _promote_dotted_field_ids({"child_id": "1.2", "task_id": "3.4"}),
            {"child_id": 1, "task_id": 4},
        )

    def test_child_id_zero_is_not_treated_as_missing(self):
        self.assertEqual(
            _promote_dotted_field_ids({"child_id": 0, "task_id": "1.2"}),
            {"child_id": 0, "task_id": 2},
        )

    def test_plain_integer_ids_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 1, "task": "Research"}
        self.assertIs(_promote_dotted_field_ids(raw), raw)

    def test_float_task_id_left_unchanged(self):
        """JSON-number 0.0 already coerces via Pydantic; do not steal #85/#94."""
        raw = {"child_id": 0, "task_id": 0.0, "task": "Research"}
        self.assertIs(_promote_dotted_field_ids(raw), raw)

    def test_dotted_id_field_left_unchanged(self):
        """Dotted `id` stays with draft PR #93."""
        raw = {"id": "0.0", "task": "Research"}
        self.assertIs(_promote_dotted_field_ids(raw), raw)

    def test_numeric_id_field_left_unchanged(self):
        """Numeric `id` stays with draft PR #94."""
        raw = {"id": 0.0, "task": "Research"}
        self.assertIs(_promote_dotted_field_ids(raw), raw)

    def test_worker_id_left_unchanged(self):
        """worker_id stays with draft PR #96."""
        raw = {"worker_id": 0, "task_id": 0, "task": "Research"}
        self.assertIs(_promote_dotted_field_ids(raw), raw)

    def test_null_child_id_not_filled_from_task_citation(self):
        raw = {"child_id": None, "task_id": "0.0", "task": "Research"}
        self.assertEqual(
            _promote_dotted_field_ids(raw),
            {"child_id": None, "task_id": 0, "task": "Research"},
        )

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_promote_dotted_field_ids("0.0"), "0.0")


class DottedFieldIdTaskRecordTests(unittest.TestCase):
    def test_task_with_dotted_task_id_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task_id": "0.0",
                "task": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_dotted_task_id(self):
        """Plan JSON `{"task_id": "0.0"}` is valid JSON then crashed on TaskRecord."""
        payload = json.loads(
            '{"child_id": 0, "task_id": "0.0",'
            ' "task": "Look up Sergey Brin\'s age", "dependencies": []}'
        )
        record = TaskRecord(**payload)
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.id, "0.0")

    def test_json_loads_bracketed_task_id(self):
        payload = json.loads(
            '{"child_id": 0, "task_id": "[0.0]", "task": "Research AGI"}'
        )
        record = TaskRecord(**payload)
        self.assertEqual(record.id, "0.0")

    def test_dependency_object_with_dotted_task_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": "0.0"}],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_child_id_citation_constructs(self):
        record = TaskRecord(
            **{"child_id": "2.0", "task_id": 1, "task": "Write the report"}
        )
        self.assertEqual(record.child_id, 2)
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.id, "2.1")

    def test_canonical_integer_ids_still_work(self):
        record = TaskRecord(
            task_id=0,
            child_id=1,
            task="Research AGI",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.id, "1.0")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_parent_plan_comprehension_accepts_dotted_task_id(self):
        plan_json = [
            {
                "child_id": 0,
                "task_id": "0.0",
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "child_id": 1,
                "task_id": "[1.0]",
                "task": "Write the report",
                "dependencies": [{"child_id": 0, "task_id": "[0.0]"}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[1].id, "1.0")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_dotted_id_still_rejected(self):
        """Dotted `id` stays with draft PR #93."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0.0", "task": "Research AGI"})

    def test_numeric_id_still_rejected(self):
        """Numeric `id` stays with draft PR #94."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0.0, "task": "Research AGI"})

    def test_worker_id_still_rejected(self):
        """worker_id stays with draft PR #96."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"worker_id": 0, "task_id": 0, "task": "Research AGI"})

    def test_integer_id_without_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0, "task": "Research AGI"})

    def test_null_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{"child_id": None, "task_id": "0.0", "task": "Research AGI"}
            )

    def test_non_citation_task_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{"child_id": 0, "task_id": "task-0", "task": "Research AGI"}
            )


class DottedFieldIdTaskRefTests(unittest.TestCase):
    def test_task_ref_from_dotted_task_id(self):
        ref = TaskRef(**{"child_id": 42, "task_id": "42.7"})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 7)
        self.assertEqual(ref.id, "42.7")

    def test_task_ref_from_dotted_child_id(self):
        ref = TaskRef(**{"child_id": "[42.7]", "task_id": 7})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 7)
        self.assertEqual(ref.id, "42.7")


if __name__ == "__main__":
    unittest.main()
