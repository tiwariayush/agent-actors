import importlib.util
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
_promote_combined_id = models._promote_combined_id


class PromoteCombinedIdTests(unittest.TestCase):
    def test_dotted_string_fills_missing_ids(self):
        self.assertEqual(
            _promote_combined_id({"id": "1.2", "task": "Research AGI"}),
            {"task": "Research AGI", "child_id": 1, "task_id": 2},
        )

    def test_bracketed_string_matches_prompt_citation(self):
        self.assertEqual(
            _promote_combined_id({"id": "[0.0]"}),
            {"child_id": 0, "task_id": 0},
        )

    def test_does_not_overwrite_explicit_ids(self):
        raw = {"id": "9.9", "child_id": 0, "task_id": 1, "task": "Keep"}
        self.assertIs(_promote_combined_id(raw), raw)

    def test_fills_only_the_missing_half(self):
        self.assertEqual(
            _promote_combined_id({"id": "2.3", "child_id": 2}),
            {"child_id": 2, "task_id": 3},
        )

    def test_non_dotted_id_left_unchanged(self):
        raw = {"id": "task-0", "task": "Research"}
        self.assertIs(_promote_combined_id(raw), raw)

    def test_integer_id_left_unchanged(self):
        raw = {"id": 0, "child_id": 0, "task": "Research"}
        self.assertIs(_promote_combined_id(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_promote_combined_id("0.0"), "0.0")


class CombinedIdTaskRecordTests(unittest.TestCase):
    def test_task_with_dotted_id_constructs(self):
        record = TaskRecord(**{"id": "0.0", "task": "Research AGI", "dependencies": []})
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Research AGI")
        self.assertEqual(record.id, "0.0")

    def test_task_with_bracketed_id_constructs(self):
        record = TaskRecord(**{"id": "[1.0]", "task": "Write the report"})
        self.assertEqual(record.child_id, 1)
        self.assertEqual(record.task_id, 0)

    def test_dependency_object_with_dotted_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"id": "0.0"}],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_canonical_fields_still_work(self):
        record = TaskRecord(
            task_id=0,
            child_id=1,
            task="Research AGI",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.id, "1.0")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_explicit_ids_win_over_conflicting_id_field(self):
        record = TaskRecord(
            **{
                "id": "9.9",
                "child_id": 0,
                "task_id": 1,
                "task": "Keep assigned ids",
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 1)

    def test_parent_plan_comprehension_accepts_id_field(self):
        plan_json = [
            {"id": "0.0", "task": "Research AGI", "dependencies": []},
            {
                "id": "[1.0]",
                "task": "Write the report",
                "dependencies": [{"id": "0.0"}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[1].id, "1.0")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_invalid_id_string_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "not-an-id", "task": "Research AGI"})

    def test_integer_id_without_task_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0, "child_id": 0, "task": "Research AGI"})


class CombinedIdTaskRefTests(unittest.TestCase):
    def test_task_ref_from_dotted_id(self):
        ref = TaskRef(**{"id": "42.7"})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 7)


if __name__ == "__main__":
    unittest.main()
