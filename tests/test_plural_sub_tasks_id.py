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
_promote_plural_sub_tasks_id = models._promote_plural_sub_tasks_id


class PromotePluralSubTasksIdTests(unittest.TestCase):
    def test_plural_sub_tasks_id_fills_missing_task_id(self):
        self.assertEqual(
            _promote_plural_sub_tasks_id(
                {
                    "child_id": 0,
                    "sub-tasks_id": 0,
                    "task": "Look up Sergey Brin's age",
                }
            ),
            {
                "child_id": 0,
                "task_id": 0,
                "task": "Look up Sergey Brin's age",
            },
        )

    def test_nonzero_plural_sub_tasks_id(self):
        self.assertEqual(
            _promote_plural_sub_tasks_id({"child_id": 0, "sub-tasks_id": 1}),
            {"child_id": 0, "task_id": 1},
        )

    def test_does_not_mutate_input(self):
        raw = {"child_id": 0, "sub-tasks_id": 0, "task": "Research AGI"}
        promoted = _promote_plural_sub_tasks_id(raw)
        self.assertEqual(raw["sub-tasks_id"], 0)
        self.assertNotIn("task_id", raw)
        self.assertEqual(promoted["task_id"], 0)
        self.assertNotIn("sub-tasks_id", promoted)

    def test_does_not_overwrite_explicit_task_id(self):
        raw = {
            "child_id": 0,
            "task_id": 1,
            "sub-tasks_id": 9,
            "task": "Keep assigned ids",
        }
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_task_id_zero_is_not_treated_as_missing(self):
        raw = {"child_id": 0, "task_id": 0, "sub-tasks_id": 9}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_null_task_id_is_replaced_by_alias(self):
        self.assertEqual(
            _promote_plural_sub_tasks_id(
                {"child_id": 0, "task_id": None, "sub-tasks_id": 1}
            ),
            {"child_id": 0, "task_id": 1},
        )

    def test_missing_alias_left_unchanged(self):
        """Bare omitted task_id stays with draft PR #100."""
        raw = {"child_id": 0, "task": "Research"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_null_plural_sub_tasks_id_left_unchanged(self):
        raw = {"child_id": 0, "sub-tasks_id": None, "task": "Research"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_bool_plural_sub_tasks_id_left_unchanged(self):
        raw = {"child_id": 0, "sub-tasks_id": False, "task": "Research"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_true_plural_sub_tasks_id_left_unchanged(self):
        raw = {"child_id": 0, "sub-tasks_id": True, "task": "Research"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_promote_plural_sub_tasks_id(0), 0)
        self.assertIsNone(_promote_plural_sub_tasks_id(None))

    def test_singular_hyphenated_sub_task_id_left_unchanged(self):
        """Singular sub-task_id stays with draft PR #114."""
        raw = {"child_id": 0, "sub-task_id": 0, "task": "Research AGI"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_fully_hyphenated_sub_task_id_left_unchanged(self):
        raw = {"child_id": 0, "sub-task-id": 0, "task": "Research AGI"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_underscore_sub_task_id_left_unchanged(self):
        """Underscore sub_task_id stays with draft PR #111."""
        raw = {"child_id": 0, "sub_task_id": 0, "task": "Research AGI"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_subtask_id_left_unchanged(self):
        """Concatenated subtask_id stays with draft PR #111."""
        raw = {"child_id": 0, "subtask_id": 0, "task": "Research AGI"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_plural_sub_tasks_text_left_unchanged(self):
        """Plural sub-tasks text stays with draft PR #115."""
        raw = {"child_id": 0, "sub-tasks": "Research AGI"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_dotted_id_field_left_unchanged(self):
        """Dotted `id` stays with draft PR #93."""
        raw = {"id": "0.0", "child_id": 0, "sub-tasks_id": 0, "task": "Research"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_numeric_id_field_left_unchanged(self):
        """Numeric `id` stays with draft PR #94."""
        raw = {"id": 0.0, "child_id": 0, "sub-tasks_id": 1, "task": "Research"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_quoted_digit_id_field_left_unchanged(self):
        """Quoted digit `id` stays with draft PR #99."""
        raw = {"id": "0", "child_id": 0, "sub-tasks_id": 0, "task": "Research"}
        self.assertIs(_promote_plural_sub_tasks_id(raw), raw)

    def test_plural_wins_when_singular_alias_also_present(self):
        self.assertEqual(
            _promote_plural_sub_tasks_id(
                {
                    "child_id": 0,
                    "sub-tasks_id": 1,
                    "sub-task_id": 9,
                    "task": "Research AGI",
                }
            ),
            {
                "child_id": 0,
                "task_id": 1,
                "sub-task_id": 9,
                "task": "Research AGI",
            },
        )


class PluralSubTasksIdRecordTests(unittest.TestCase):
    def test_task_with_plural_sub_tasks_id_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "sub-tasks_id": 0,
                "task": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_nonzero_plural_sub_tasks_id_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "sub-tasks_id": 1,
                "task": "Synthesize findings",
                "dependencies": [{"child_id": 0, "sub-tasks_id": 0}],
            }
        )
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.id, "0.1")
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_json_loads_plural_sub_tasks_id(self):
        """Plan JSON `{"sub-tasks_id": 0}` is valid JSON then crashed on TaskRecord."""
        payload = json.loads(
            '{"child_id": 0, "sub-tasks_id": 0,'
            ' "task": "Look up Sergey Brin\'s age", "dependencies": []}'
        )
        self.assertNotIn("task_id", payload)
        self.assertEqual(payload["sub-tasks_id"], 0)
        record = TaskRecord(**payload)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.id, "0.0")
        self.assertEqual(record.task, "Look up Sergey Brin's age")

    def test_dependency_object_with_plural_sub_tasks_id(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "sub-tasks_id": 0}],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
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

    def test_explicit_task_id_wins_over_conflicting_alias(self):
        record = TaskRecord(
            **{
                "sub-tasks_id": 9,
                "child_id": 0,
                "task_id": 1,
                "task": "Keep assigned ids",
            }
        )
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.child_id, 0)

    def test_parent_plan_comprehension_accepts_plural_sub_tasks_id(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = json.loads(
            """
            [
              {
                "child_id": 0,
                "sub-tasks_id": 0,
                "task": "Research AGI",
                "dependencies": []
              },
              {
                "child_id": 2,
                "sub-tasks_id": 0,
                "task": "Write the executive report",
                "dependencies": [{"child_id": 0, "sub-tasks_id": 0}]
              }
            ]
            """
        )
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].task, "Write the executive report")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_string_plural_sub_tasks_id_coerces(self):
        record = TaskRecord(
            **{"child_id": 2, "sub-tasks_id": "0", "task": "Write the report"}
        )
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.child_id, 2)

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

    def test_singular_hyphenated_sub_task_id_still_rejected(self):
        """Singular sub-task_id stays with draft PR #114."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "sub-task_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_fully_hyphenated_sub_task_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "sub-task-id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_underscore_sub_task_id_still_rejected(self):
        """Underscore sub_task_id stays with draft PR #111."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "sub_task_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_subtask_id_still_rejected(self):
        """Concatenated subtask_id stays with draft PR #111."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "subtask_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_plural_sub_tasks_text_still_rejected(self):
        """Plural sub-tasks text stays with draft PR #115; this PR does not fill task."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub-tasks": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_plural_id_without_task_text_still_rejected(self):
        """Filling task_id must not invent a task string (#115 / #103)."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "sub-tasks_id": 0,
                    "sub-tasks": "Research AGI",
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

    def test_null_plural_sub_tasks_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "sub-tasks_id": None,
                    "task": "Research AGI",
                }
            )

    def test_bool_plural_sub_tasks_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "sub-tasks_id": True,
                    "task": "Research AGI",
                }
            )

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "sub-tasks_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_dependency_singular_alias_still_rejected(self):
        """Dependency sub-task_id stays with draft PR #114."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Synthesize",
                dependencies=[{"child_id": 0, "sub-task_id": 0}],
            )


class PluralSubTasksIdRefTests(unittest.TestCase):
    def test_task_ref_from_plural_sub_tasks_id(self):
        ref = TaskRef(**{"child_id": 42, "sub-tasks_id": 7})
        self.assertEqual(ref.child_id, 42)
        self.assertEqual(ref.task_id, 7)
        self.assertEqual(ref.id, "42.7")

    def test_task_ref_canonical_task_id_still_works(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_task_ref_missing_task_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRef(**{"child_id": 0})

    def test_task_ref_singular_alias_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRef(**{"child_id": 0, "sub-task_id": 0})

    def test_task_ref_fully_hyphenated_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRef(**{"child_id": 0, "sub-task-id": 0})


if __name__ == "__main__":
    unittest.main()
