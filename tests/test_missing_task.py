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
_fill_missing_task = models._fill_missing_task


class FillMissingTaskTests(unittest.TestCase):
    def test_description_sibling(self):
        self.assertEqual(
            _fill_missing_task(
                {"child_id": 0, "task_id": 0, "description": "Research AGI"}
            )["task"],
            "Research AGI",
        )

    def test_text_sibling(self):
        self.assertEqual(
            _fill_missing_task(
                {"child_id": 0, "task_id": 0, "text": "Write the executive report"}
            )["task"],
            "Write the executive report",
        )

    def test_objective_sibling(self):
        self.assertEqual(
            _fill_missing_task(
                {"child_id": 0, "task_id": 0, "objective": "Synthesize findings"}
            )["task"],
            "Synthesize findings",
        )

    def test_instruction_sibling(self):
        self.assertEqual(
            _fill_missing_task(
                {
                    "child_id": 0,
                    "task_id": 0,
                    "instruction": "Include 5 jokes and quotes",
                }
            )["task"],
            "Include 5 jokes and quotes",
        )

    def test_prefers_description_over_text(self):
        self.assertEqual(
            _fill_missing_task(
                {
                    "child_id": 0,
                    "task_id": 0,
                    "description": "Research AGI",
                    "text": "ignored extra",
                }
            )["task"],
            "Research AGI",
        )

    def test_skips_empty_description_for_later_sibling(self):
        self.assertEqual(
            _fill_missing_task(
                {
                    "child_id": 0,
                    "task_id": 0,
                    "description": "",
                    "objective": "Synthesize findings",
                }
            )["task"],
            "Synthesize findings",
        )

    def test_omitted_task_defaults_to_empty_string(self):
        self.assertEqual(
            _fill_missing_task({"child_id": 0, "task_id": 0, "dependencies": []})[
                "task"
            ],
            "",
        )

    def test_existing_task_left_unchanged(self):
        raw = {
            "child_id": 0,
            "task_id": 0,
            "task": "Research AGI",
            "description": "ignored extra",
        }
        self.assertIs(_fill_missing_task(raw), raw)

    def test_null_task_left_unchanged(self):
        """Null task stays with the existing null-task repair."""
        raw = {"child_id": 0, "task_id": 0, "task": None}
        self.assertIs(_fill_missing_task(raw), raw)

    def test_list_task_left_unchanged(self):
        """List-valued task stays with the existing list-task repair."""
        raw = {"child_id": 0, "task_id": 0, "task": ["Research AGI", "Write the report"]}
        self.assertIs(_fill_missing_task(raw), raw)

    def test_object_task_left_unchanged(self):
        """Object-shaped task stays with the existing object-task repair."""
        raw = {"child_id": 0, "task_id": 0, "task": {"description": "Research AGI"}}
        self.assertIs(_fill_missing_task(raw), raw)

    def test_non_mapping_left_unchanged(self):
        raw = [{"child_id": 0}]
        self.assertIs(_fill_missing_task(raw), raw)


class MissingTaskRecordTests(unittest.TestCase):
    def test_omitted_task_constructs_empty_string(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task_id": 0,
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_omitted_task(self):
        """Plan JSON that omits `task` is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0, "dependencies": []}'
        )
        self.assertNotIn("task", payload)
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_description_sibling_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task_id": 0,
                "description": "Look up Sergey Brin's age",
                "dependencies": [],
            }
        )
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_description_instead_of_task(self):
        payload = json.loads(
            '{"child_id": 2, "task_id": 0,'
            ' "description": "Write the executive report",'
            ' "dependencies": []}'
        )
        self.assertNotIn("task", payload)
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "Write the executive report")
        self.assertEqual(record.child_id, 2)

    def test_canonical_string_task_still_works(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task, "Synthesize")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_parent_plan_comprehension_accepts_omitted_task(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "description": "Research AGI",
                "dependencies": [],
            },
            {
                "child_id": 2,
                "task_id": 0,
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "Research AGI")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].task, "")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_empty_string_task_still_works(self):
        record = TaskRecord(
            child_id=0,
            task_id=0,
            task="",
            dependencies=[],
        )
        self.assertEqual(record.task, "")

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

    def test_object_task_still_rejected(self):
        """Object-shaped task remains a separate object-task repair."""
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
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {},
                    "dependencies": [],
                }
            )

    def test_unknown_key_object_task_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"notes": "Research AGI"},
                    "dependencies": [],
                }
            )

    def test_non_string_description_defaults_empty(self):
        record = TaskRecord(
            **{
                "child_id": 0,
                "task_id": 0,
                "description": {"text": "Research AGI"},
                "dependencies": [],
            }
        )
        self.assertEqual(record.task, "")

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "task_id": 0,
                    "description": "Research AGI",
                    "dependencies": [],
                }
            )


class MissingTaskRefTests(unittest.TestCase):
    def test_task_ref_does_not_require_task_field(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_dependency_object_ignores_missing_task(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Write the report",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.task, "Write the report")


if __name__ == "__main__":
    unittest.main()
