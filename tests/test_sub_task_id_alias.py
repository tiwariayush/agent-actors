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
_promote_sub_task_id = models._promote_sub_task_id


class PromoteSubTaskIdTests(unittest.TestCase):
    def test_sub_task_id_fills_missing_task_id(self):
        self.assertEqual(
            _promote_sub_task_id(
                {"child_id": 0, "sub_task_id": 0, "task": "Research AGI"}
            ),
            {"child_id": 0, "task_id": 0, "task": "Research AGI"},
        )

    def test_subtask_id_fills_missing_task_id(self):
        self.assertEqual(
            _promote_sub_task_id(
                {
                    "child_id": 2,
                    "subtask_id": 0,
                    "task": "Write the executive report",
                }
            ),
            {
                "child_id": 2,
                "task_id": 0,
                "task": "Write the executive report",
            },
        )

    def test_nonzero_sub_task_id(self):
        self.assertEqual(
            _promote_sub_task_id({"child_id": 0, "sub_task_id": 1}),
            {"child_id": 0, "task_id": 1},
        )

    def test_prefers_sub_task_id_over_subtask_id(self):
        self.assertEqual(
            _promote_sub_task_id(
                {"child_id": 0, "sub_task_id": 0, "subtask_id": 9}
            ),
            {"child_id": 0, "task_id": 0, "subtask_id": 9},
        )

    def test_does_not_overwrite_explicit_task_id(self):
        raw = {
            "child_id": 0,
            "task_id": 1,
            "sub_task_id": 9,
            "task": "Keep assigned ids",
        }
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_task_id_zero_is_not_treated_as_missing(self):
        raw = {"child_id": 0, "task_id": 0, "sub_task_id": 9}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_subtask_id_does_not_overwrite_task_id(self):
        raw = {"child_id": 2, "task_id": 0, "subtask_id": 9}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_null_task_id_is_replaced_by_alias(self):
        self.assertEqual(
            _promote_sub_task_id(
                {"child_id": 0, "task_id": None, "sub_task_id": 1}
            ),
            {"child_id": 0, "task_id": 1},
        )

    def test_missing_aliases_left_unchanged(self):
        """Bare omitted task_id stays with draft PR #100."""
        raw = {"child_id": 0, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_null_sub_task_id_left_unchanged(self):
        raw = {"child_id": 0, "sub_task_id": None, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_null_subtask_id_left_unchanged(self):
        raw = {"child_id": 0, "subtask_id": None, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_bool_sub_task_id_left_unchanged(self):
        raw = {"child_id": 0, "sub_task_id": False, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_bool_subtask_id_left_unchanged(self):
        raw = {"child_id": 0, "subtask_id": True, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_promote_sub_task_id(0), 0)

    def test_dotted_id_field_left_unchanged(self):
        """Dotted `id` stays with draft PR #93."""
        raw = {"id": "0.0", "child_id": 0, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_numeric_id_field_left_unchanged(self):
        """Numeric `id` stays with draft PR #94."""
        raw = {"id": 0.0, "child_id": 0, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_quoted_digit_id_field_left_unchanged(self):
        """Quoted digit `id` stays with draft PR #99."""
        raw = {"id": "0", "child_id": 0, "task": "Research"}
        self.assertIs(_promote_sub_task_id(raw), raw)

    def test_sub_task_text_alias_left_unchanged(self):
        """sub_task text stays with draft PR #110; this helper only fills task_id."""
        raw = {"child_id": 0, "sub_task": "Research AGI"}
        self.assertIs(_promote_sub_task_id(raw), raw)


class SubTaskIdRecordTests(unittest.TestCase):
    def test_task_with_sub_task_id_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "sub_task_id": 0,
                "task": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_task_with_subtask_id_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 42,
                "subtask_id": 0,
                "task": "Draft jokes and quotes",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 42)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.id, "42.0")

    def test_nonzero_sub_task_id_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "sub_task_id": 1,
                "task": "Synthesize findings",
                "dependencies": [{"child_id": 0, "sub_task_id": 0}],
            }
        )
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.id, "0.1")
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_json_loads_sub_task_id(self):
        """Plan JSON `{"sub_task_id": 0}` is valid JSON then crashed on TaskRecord."""
        payload = json.loads(
            '{"child_id": 0, "sub_task_id": 0,'
            ' "task": "Look up Sergey Brin\'s age", "dependencies": []}'
        )
        self.assertNotIn("task_id", payload)
        record = TaskRecord(**payload)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.id, "0.0")

    def test_json_loads_subtask_id(self):
        payload = json.loads(
            '{"child_id": 2, "subtask_id": 0,'
            ' "task": "Write the executive report", "dependencies": []}'
        )
        self.assertNotIn("task_id", payload)
        record = TaskRecord(**payload)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.child_id, 2)

    def test_dependency_object_with_sub_task_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "sub_task_id": 0}],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_dependency_object_with_subtask_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=2,
            task="Write the report",
            dependencies=[{"child_id": 0, "subtask_id": 0}],
        )
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_canonical_task_id_still_works(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_explicit_task_id_wins_over_conflicting_sub_task_id(self):
        record = TaskRecord(
            **{
                "sub_task_id": 9,
                "child_id": 0,
                "task_id": 1,
                "task": "Keep assigned ids",
            }
        )
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.child_id, 0)

    def test_parent_plan_comprehension_accepts_sub_task_id(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "sub_task_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "child_id": 1,
                "subtask_id": 0,
                "task": "Write the report",
                "dependencies": [{"child_id": 0, "sub_task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")
        self.assertEqual(records[1].id, "1.0")
        self.assertEqual(records[1].task, "Write the report")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_string_sub_task_id_coerces(self):
        record = TaskRecord(
            **{"child_id": 2, "sub_task_id": "0", "task": "Write the report"}
        )
        self.assertEqual(record.task_id, 0)

    def test_string_subtask_id_coerces(self):
        record = TaskRecord(
            **{"child_id": 42, "subtask_id": "1", "task": "Draft jokes"}
        )
        self.assertEqual(record.task_id, 1)

    def test_missing_task_id_without_alias_still_rejected(self):
        """Bare omitted task_id stays with draft PR #100."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_null_task_id_without_alias_still_rejected(self):
        """JSON-null task_id stays with draft PR #100."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": None,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_dotted_id_still_rejected(self):
        """Dotted `id` stays with draft PR #93."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0.0", "child_id": 0, "task": "Research AGI"})

    def test_numeric_id_still_rejected(self):
        """Numeric `id` stays with draft PR #94."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0.0, "child_id": 0, "task": "Research AGI"})

    def test_quoted_digit_id_still_rejected(self):
        """Quoted digit `id` stays with draft PR #99."""
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": "0", "child_id": 0, "task": "Research AGI"})

    def test_integer_id_without_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(**{"id": 0, "task": "Research AGI"})

    def test_null_sub_task_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "sub_task_id": None,
                    "task": "Research AGI",
                }
            )

    def test_sub_task_text_alias_still_rejected(self):
        """sub_task text stays with draft PR #110; this PR does not fill task."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub_task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "sub_task_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )


class SubTaskIdRefTests(unittest.TestCase):
    def test_task_ref_from_sub_task_id(self):
        ref = TaskRef(**{"child_id": 42, "sub_task_id": 7})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 7)
        self.assertEqual(ref.id, "42.7")

    def test_task_ref_from_subtask_id(self):
        ref = TaskRef(**{"child_id": 2, "subtask_id": 0})
        self.assertEqual(ref.child_id, 2)
        self.assertEqual(ref.task_id, 0)
        self.assertEqual(ref.id, "2.0")

    def test_task_ref_canonical_task_id_still_works(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_task_ref_missing_task_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRef(**{"child_id": 0})


if __name__ == "__main__":
    unittest.main()
