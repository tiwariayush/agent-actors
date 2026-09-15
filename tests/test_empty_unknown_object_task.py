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
_task_from_empty_or_unknown_object = models._task_from_empty_or_unknown_object


class TaskFromEmptyOrUnknownObjectTests(unittest.TestCase):
    def test_empty_object_becomes_empty_string(self):
        self.assertEqual(_task_from_empty_or_unknown_object({}), "")

    def test_title_key(self):
        self.assertEqual(
            _task_from_empty_or_unknown_object({"title": "Research AGI"}),
            "Research AGI",
        )

    def test_notes_key(self):
        """Unknown-key objects were left untouched by the known-key extractor."""
        self.assertEqual(
            _task_from_empty_or_unknown_object({"notes": "Research AGI"}),
            "Research AGI",
        )

    def test_summary_key(self):
        self.assertEqual(
            _task_from_empty_or_unknown_object({"summary": "Write the executive report"}),
            "Write the executive report",
        )

    def test_name_key(self):
        self.assertEqual(
            _task_from_empty_or_unknown_object({"name": "Look up Sergey Brin's age"}),
            "Look up Sergey Brin's age",
        )

    def test_content_key(self):
        self.assertEqual(
            _task_from_empty_or_unknown_object({"content": "Include 5 jokes and quotes"}),
            "Include 5 jokes and quotes",
        )

    def test_single_string_with_non_string_sibling(self):
        self.assertEqual(
            _task_from_empty_or_unknown_object({"title": "Research AGI", "priority": 1}),
            "Research AGI",
        )

    def test_plain_string_left_unchanged(self):
        raw = "Research AGI"
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_null_task_left_unchanged(self):
        """Null task stays with the existing null-task repair."""
        self.assertIsNone(_task_from_empty_or_unknown_object(None))

    def test_list_task_left_unchanged(self):
        """List-valued task stays with the existing list-task repair."""
        raw = ["Research AGI", "Write the report"]
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_empty_list_left_unchanged(self):
        raw = []
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_description_object_left_unchanged(self):
        """Known object-task keys stay with the existing object-task repair."""
        raw = {"description": "Research AGI"}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_nested_task_key_left_unchanged(self):
        raw = {"task": "Look up Sergey Brin's age"}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_text_key_left_unchanged(self):
        raw = {"text": "Write the executive report"}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_objective_key_left_unchanged(self):
        raw = {"objective": "Synthesize findings"}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_instruction_key_left_unchanged(self):
        raw = {"instruction": "Include 5 jokes and quotes"}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_known_key_preferred_even_with_title(self):
        raw = {"title": "ignored extra", "description": "Research AGI"}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_multiple_unknown_strings_left_unchanged(self):
        raw = {"title": "Research AGI", "notes": "also a candidate"}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)

    def test_unknown_non_string_only_left_unchanged(self):
        raw = {"count": 1}
        self.assertIs(_task_from_empty_or_unknown_object(raw), raw)


class EmptyUnknownObjectTaskRecordTests(unittest.TestCase):
    def test_empty_object_constructs(self):
        record = TaskRecord(
            child_id=0,
            task_id=0,
            task={},
            dependencies=[],
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 0)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_empty_object_task(self):
        """Plan JSON `\"task\": {}` is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 0, "task": {}, "dependencies": []}'
        )
        self.assertEqual(payload["task"], {})
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "")
        self.assertEqual(record.id, "0.0")

    def test_json_loads_title_object_task(self):
        payload = json.loads(
            '{"child_id": 0, "task_id": 0,'
            ' "task": {"title": "Look up Sergey Brin\'s age"},'
            ' "dependencies": []}'
        )
        self.assertIsInstance(payload["task"], dict)
        record = TaskRecord(**payload)
        self.assertEqual(record.task, "Look up Sergey Brin's age")
        self.assertEqual(record.id, "0.0")

    def test_notes_object_constructs(self):
        record = TaskRecord(
            **{
                "child_id": 2,
                "task_id": 0,
                "task": {"notes": "Write the executive report"},
                "dependencies": [],
            }
        )
        self.assertEqual(record.child_id, 2)
        self.assertEqual(record.task, "Write the executive report")
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

    def test_parent_plan_comprehension_accepts_empty_and_unknown_object_task(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": {},
                "dependencies": [],
            },
            {
                "child_id": 2,
                "task_id": 0,
                "task": {"title": "Write the executive report"},
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[0].task, "")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].task, "Write the executive report")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

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

    def test_description_object_still_rejected(self):
        """Known-key object tasks remain draft PR #102's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"description": "Research AGI"},
                    "dependencies": [],
                }
            )

    def test_nested_task_key_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"task": "Research AGI"},
                    "dependencies": [],
                }
            )

    def test_multiple_unknown_strings_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"title": "Research AGI", "notes": "also a candidate"},
                    "dependencies": [],
                }
            )

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "task_id": 0,
                    "task": {"title": "Research AGI"},
                    "dependencies": [],
                }
            )


class EmptyUnknownObjectTaskRefTests(unittest.TestCase):
    def test_task_ref_does_not_require_task_field(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")

    def test_dependency_object_ignores_unknown_object_task(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Write the report",
            dependencies=[
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": {"title": "Research AGI"},
                }
            ],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.task, "Write the report")


if __name__ == "__main__":
    unittest.main()
