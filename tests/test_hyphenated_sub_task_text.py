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
_promote_hyphenated_sub_task = models._promote_hyphenated_sub_task


class PromoteHyphenatedSubTaskTests(unittest.TestCase):
    def test_hyphenated_sub_task_fills_missing_task(self):
        self.assertEqual(
            _promote_hyphenated_sub_task(
                {"child_id": 0, "task_id": 0, "sub-task": "Research AGI"}
            ),
            {"child_id": 0, "task_id": 0, "task": "Research AGI"},
        )

    def test_does_not_overwrite_explicit_task(self):
        raw = {
            "child_id": 0,
            "task_id": 0,
            "task": "Keep assigned task",
            "sub-task": "ignored extra",
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_empty_string_task_is_not_treated_as_missing(self):
        raw = {
            "child_id": 0,
            "task_id": 0,
            "task": "",
            "sub-task": "Research",
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_null_task_left_unchanged(self):
        """Null task stays with draft PR #101."""
        raw = {
            "child_id": 0,
            "task_id": 0,
            "task": None,
            "sub-task": "Research",
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_list_task_left_unchanged(self):
        """List-valued task stays with draft PR #98."""
        raw = {
            "child_id": 0,
            "task_id": 0,
            "task": ["Research AGI", "Write the report"],
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_empty_list_task_left_unchanged(self):
        """Empty-list task stays with draft PR #108."""
        raw = {"child_id": 0, "task_id": 0, "task": []}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_object_task_left_unchanged(self):
        """Object-shaped task stays with draft PR #102."""
        raw = {
            "child_id": 0,
            "task_id": 0,
            "task": {"description": "Research AGI"},
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_empty_object_task_left_unchanged(self):
        """Empty object task stays with draft PR #107."""
        raw = {"child_id": 0, "task_id": 0, "task": {}}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_underscore_sub_task_left_unchanged(self):
        """sub_task / subtask stay with draft PR #110."""
        raw = {"child_id": 0, "task_id": 0, "sub_task": "Research AGI"}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_concatenated_subtask_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 0, "subtask": "Write the report"}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_description_sibling_left_unchanged(self):
        """description/text/objective/instruction stay with draft PR #103."""
        raw = {
            "child_id": 0,
            "task_id": 0,
            "description": "Research AGI",
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_text_sibling_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 0, "text": "Write the report"}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_objective_sibling_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 0, "objective": "Synthesize findings"}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_instruction_sibling_left_unchanged(self):
        raw = {
            "child_id": 0,
            "task_id": 0,
            "instruction": "Include 5 jokes and quotes",
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_omitted_task_without_alias_left_unchanged(self):
        """Bare omitted task stays with draft PR #103."""
        raw = {"child_id": 0, "task_id": 0, "dependencies": []}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_plural_sub_tasks_string_left_unchanged(self):
        """Plan-level hyphenated wrappers stay with draft PR #112."""
        raw = {"child_id": 0, "task_id": 0, "sub-tasks": "Research AGI"}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_null_hyphenated_sub_task_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 0, "sub-task": None}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_empty_hyphenated_sub_task_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 0, "sub-task": ""}
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_list_hyphenated_sub_task_left_unchanged(self):
        raw = {
            "child_id": 0,
            "task_id": 0,
            "sub-task": ["Research AGI", "Write the report"],
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_object_hyphenated_sub_task_left_unchanged(self):
        raw = {
            "child_id": 0,
            "task_id": 0,
            "sub-task": {"description": "Research AGI"},
        }
        self.assertIs(_promote_hyphenated_sub_task(raw), raw)

    def test_non_mapping_left_unchanged(self):
        self.assertEqual(_promote_hyphenated_sub_task(0), 0)


class HyphenatedSubTaskRecordTests(unittest.TestCase):
    def test_task_with_hyphenated_sub_task_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task_id": 0,
                "sub-task": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_hyphenated_sub_task(self):
        """Plan JSON `{"sub-task": "..."}` is valid JSON then crashed on TaskRecord."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0, "sub-task": "Look up Sergey Brin\'s age",'
            ' "dependencies": []}'
        )
        self.assertNotIn("task", payload)
        self.assertIn("sub-task", payload)
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_canonical_task_still_works(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task, "Synthesize")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_explicit_task_wins_over_conflicting_hyphenated_sub_task(self):
        record = TaskRecord(
            **{
                "sub-task": "ignored extra",
                "child_id": 0,
                "task_id": 1,
                "task": "Keep assigned task",
            }
        )
        self.assertEqual(record.task, "Keep assigned task")
        self.assertEqual(record.task_id, 1)

    def test_parent_plan_comprehension_accepts_hyphenated_sub_task(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "sub-task": "Research AGI",
                "dependencies": [],
            },
            {
                "child_id": 1,
                "task_id": 0,
                "task": "Write the report",
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")
        self.assertEqual(records[1].id, "1.0")
        self.assertEqual(records[1].task, "Write the report")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_null_task_still_rejected(self):
        """JSON-null task stays with draft PR #101."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": None,
                    "dependencies": [],
                }
            )

    def test_list_task_still_rejected(self):
        """List-valued task stays with draft PR #98."""
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
        """Empty-list task stays with draft PR #108."""
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
        """Object-shaped task stays with draft PR #102."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"description": "Research AGI"},
                    "dependencies": [],
                }
            )

    def test_empty_object_task_still_rejected(self):
        """Empty object task stays with draft PR #107."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {},
                    "dependencies": [],
                }
            )

    def test_underscore_sub_task_still_rejected(self):
        """sub_task text stays with draft PR #110."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub_task": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_concatenated_subtask_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "subtask": "Write the report",
                    "dependencies": [],
                }
            )

    def test_description_sibling_still_rejected(self):
        """description sibling stays with draft PR #103."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "description": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_omitted_task_still_rejected(self):
        """Bare omitted task stays with draft PR #103."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "dependencies": [],
                }
            )

    def test_plan_wrapper_sub_tasks_still_typeerrors(self):
        """``{"sub-tasks": [...]}`` stays with draft PR #112."""
        wrapped = {
            "sub-tasks": [
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                }
            ]
        }
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in wrapped]

    def test_plan_wrapper_sub_task_list_still_typeerrors(self):
        """Singular plan-level wrapper stays with draft PR #112."""
        wrapped = {
            "sub-task": [
                {
                    "child_id": 2,
                    "task_id": 0,
                    "task": "Write the report",
                    "dependencies": [],
                }
            ]
        }
        with self.assertRaises(TypeError):
            [TaskRecord(**t) for t in wrapped]

    def test_plural_sub_tasks_string_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub-tasks": "Research AGI",
                    "dependencies": [],
                }
            )

    def test_null_hyphenated_sub_task_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub-task": None,
                    "dependencies": [],
                }
            )

    def test_list_hyphenated_sub_task_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "sub-task": ["Research AGI", "Write the report"],
                    "dependencies": [],
                }
            )

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "task_id": 0,
                    "sub-task": "Research AGI",
                    "dependencies": [],
                }
            )


class HyphenatedSubTaskRefTests(unittest.TestCase):
    def test_task_ref_does_not_require_task_field(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_dependency_object_ignores_hyphenated_sub_task(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Write the report",
            dependencies=[
                {"child_id": 0, "task_id": 0, "sub-task": "Research AGI"}
            ],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.task, "Write the report")


if __name__ == "__main__":
    unittest.main()
