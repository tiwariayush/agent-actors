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
_promote_digit_string_id = models._promote_digit_string_id


class PromoteDigitStringIdTests(unittest.TestCase):
    def test_quoted_digit_aliases_task_id_when_child_id_present(self):
        self.assertEqual(
            _promote_digit_string_id(
                {"id": "0", "child_id": 1, "task": "Research AGI"}
            ),
            {"child_id": 1, "task": "Research AGI", "task_id": 0},
        )

    def test_nonzero_quoted_digit_aliases_task_id(self):
        self.assertEqual(
            _promote_digit_string_id({"id": "12", "child_id": 0}),
            {"child_id": 0, "task_id": 12},
        )

    def test_strips_whitespace_around_digits(self):
        self.assertEqual(
            _promote_digit_string_id({"id": " 3 ", "child_id": 2}),
            {"child_id": 2, "task_id": 3},
        )

    def test_does_not_overwrite_explicit_task_id(self):
        raw = {"id": "9", "child_id": 0, "task_id": 1, "task": "Keep"}
        self.assertIs(_promote_digit_string_id(raw), raw)

    def test_quoted_digit_without_child_id_left_unchanged(self):
        raw = {"id": "0", "task": "Research AGI"}
        self.assertIs(_promote_digit_string_id(raw), raw)

    def test_dotted_string_id_left_unchanged(self):
        """Dotted citations stay with the existing combined-id repair."""
        raw = {"id": "0.0", "task": "Research AGI"}
        self.assertIs(_promote_digit_string_id(raw), raw)

    def test_bracketed_citation_left_unchanged(self):
        raw = {"id": "[0.0]", "child_id": 0, "task": "Research AGI"}
        self.assertIs(_promote_digit_string_id(raw), raw)

    def test_integer_id_left_unchanged(self):
        """Unquoted integer id stays with the existing numeric-id repair."""
        raw = {"id": 0, "child_id": 0, "task": "Research AGI"}
        self.assertIs(_promote_digit_string_id(raw), raw)

    def test_float_id_left_unchanged(self):
        raw = {"id": 0.0, "task": "Research AGI"}
        self.assertIs(_promote_digit_string_id(raw), raw)

    def test_non_digit_string_left_unchanged(self):
        raw = {"id": "task-0", "child_id": 0, "task": "Research"}
        self.assertIs(_promote_digit_string_id(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_promote_digit_string_id("0"), "0")


class DigitStringIdTaskRecordTests(unittest.TestCase):
    def test_task_with_quoted_digit_id_and_child_id_constructs(self):
        record = TaskRecord(
            **{
                "id": "0",
                "child_id": 0,
                "task": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_quoted_digit_id(self):
        """Plan JSON `{"id": "0"}` stays a string after json.loads."""
        payload = json.loads(
            '{"id": "0", "child_id": 0, "task": "Research AGI", "dependencies": []}'
        )
        self.assertIsInstance(payload["id"], str)
        record = TaskRecord(**payload)
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.id, "0.0")

    def test_dependency_object_with_quoted_digit_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"id": "0", "child_id": 0}],
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

    def test_explicit_task_id_wins_over_conflicting_quoted_id(self):
        record = TaskRecord(
            **{
                "id": "9",
                "child_id": 0,
                "task_id": 1,
                "task": "Keep assigned ids",
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 1)

    def test_parent_plan_comprehension_accepts_quoted_digit_id(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "id": "0",
                "child_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "id": "1",
                "child_id": 0,
                "task": "Write the report",
                "dependencies": [{"id": "0", "child_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[1].id, "0.1")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_dotted_string_id_still_rejected(self):
        """\"id\": \"0.0\" remains a separate combined-id repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0.0", "task": "Research AGI"})

    def test_integer_id_with_child_id_still_rejected(self):
        """Unquoted integer id remains a separate numeric-id repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0, "child_id": 0, "task": "Research AGI"})

    def test_quoted_digit_without_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0", "task": "Research AGI"})

    def test_float_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0.0, "task": "Research AGI"})


class DigitStringIdTaskRefTests(unittest.TestCase):
    def test_task_ref_from_quoted_digit_id_with_child_id(self):
        ref = TaskRef(**{"id": "7", "child_id": 42})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 7)


if __name__ == "__main__":
    unittest.main()
