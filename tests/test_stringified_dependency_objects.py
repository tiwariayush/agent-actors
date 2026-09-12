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
_stringified_dependency_object_list = models._stringified_dependency_object_list


class StringifiedDependencyObjectListTests(unittest.TestCase):
    def test_stringified_single_object_array(self):
        self.assertEqual(
            _stringified_dependency_object_list('[{"child_id": 0, "task_id": 0}]'),
            [{"child_id": 0, "task_id": 0}],
        )

    def test_whitespace_padded_stringified_array(self):
        self.assertEqual(
            _stringified_dependency_object_list(
                ' [ { "child_id": 0, "task_id": 0 } ] '
            ),
            [{"child_id": 0, "task_id": 0}],
        )

    def test_stringified_multi_object_array(self):
        self.assertEqual(
            _stringified_dependency_object_list(
                '[{"child_id": 0, "task_id": 0}, {"child_id": 1, "task_id": 0}]'
            ),
            [
                {"child_id": 0, "task_id": 0},
                {"child_id": 1, "task_id": 0},
            ],
        )

    def test_canonical_list_left_unchanged(self):
        raw = [{"child_id": 0, "task_id": 0}]
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_empty_array_string_left_unchanged(self):
        """Empty encodings stay with the existing empty-string repair."""
        raw = "[]"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_whitespace_empty_array_string_left_unchanged(self):
        raw = " [ ] "
        self.assertEqual(_stringified_dependency_object_list(raw), raw)

    def test_empty_string_left_unchanged(self):
        raw = ""
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_stringified_null_left_unchanged(self):
        raw = "null"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_stringified_empty_object_left_unchanged(self):
        raw = "{}"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_citation_string_left_unchanged(self):
        """Quoted task-ref strings stay with the existing citation repair."""
        raw = "0.0"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_bracketed_citation_string_left_unchanged(self):
        raw = "[0.0]"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_comma_joined_citations_left_unchanged(self):
        raw = "0.0, 1.0"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_integer_pair_string_left_unchanged(self):
        """Stringified [worker, task] pairs stay with the pair repair."""
        raw = "[0, 0]"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_null_item_string_left_unchanged(self):
        raw = "[null]"
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_mixed_object_and_null_string_left_unchanged(self):
        raw = '[{"child_id": 0, "task_id": 0}, null]'
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_stringified_single_object_left_unchanged(self):
        """A stringified bare object stays with the bare-object repair."""
        raw = '{"child_id": 0, "task_id": 0}'
        self.assertIs(_stringified_dependency_object_list(raw), raw)

    def test_null_left_unchanged(self):
        self.assertIsNone(_stringified_dependency_object_list(None))

    def test_bare_object_left_unchanged(self):
        raw = {"child_id": 0, "task_id": 0}
        self.assertIs(_stringified_dependency_object_list(raw), raw)


class StringifiedDependencyObjectRecordTests(unittest.TestCase):
    def test_stringified_array_constructs(self):
        record = TaskRecord(
            child_id=0,
            task_id=1,
            task="Write the executive report",
            dependencies='[{"child_id": 0, "task_id": 0}]',
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.task, "Write the executive report")
        self.assertEqual(record.id, "0.1")
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_json_loads_stringified_array(self):
        """Plan JSON that stringifies a dependency array is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 1,'
            ' "task": "Write the executive report",'
            ' "dependencies": "[{\\"child_id\\": 0, \\"task_id\\": 0}]"}'
        )
        self.assertIsInstance(payload["dependencies"], str)
        record = TaskRecord(**payload)
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.id, "0.1")

    def test_stringified_multi_object_array_constructs(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Synthesize findings",
            dependencies=(
                '[{"child_id": 0, "task_id": 0},'
                ' {"child_id": 1, "task_id": 0}]'
            ),
        )
        self.assertEqual([dep.id for dep in record.dependencies], ["0.0", "1.0"])
        self.assertEqual(record.id, "2.0")

    def test_canonical_list_still_works(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task, "Synthesize")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_parent_plan_comprehension_accepts_stringified_array(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "child_id": 2,
                "task_id": 0,
                "task": "Write the executive report",
                "dependencies": '[{"child_id": 0, "task_id": 0}]',
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_empty_array_string_still_rejected(self):
        """Empty encodings remain a separate empty-string repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=0,
                task="Research AGI",
                dependencies="[]",
            )

    def test_citation_string_still_rejected(self):
        """Quoted task-ref strings remain a separate citation repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies="0.0",
            )

    def test_bracketed_citation_string_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies="[0.0]",
            )

    def test_comma_joined_citations_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=2,
                task_id=0,
                task="Synthesize",
                dependencies="0.0, 1.0",
            )

    def test_integer_pair_string_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies="[0, 0]",
            )

    def test_stringified_null_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=0,
                task="Research AGI",
                dependencies="null",
            )

    def test_bare_object_still_rejected(self):
        """A non-string dependency object remains a separate wrap repair."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies={"child_id": 0, "task_id": 0},
            )

    def test_missing_child_id_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                **{
                    "task_id": 1,
                    "task": "Write the report",
                    "dependencies": '[{"child_id": 0, "task_id": 0}]',
                }
            )


class StringifiedDependencyObjectRefTests(unittest.TestCase):
    def test_task_ref_does_not_accept_dependencies(self):
        ref = TaskRef(child_id=0, task_id=1)
        self.assertEqual(ref.id, "0.1")
        self.assertFalse(hasattr(ref, "dependencies") and ref.dependencies)

    def test_canonical_dependency_objects_still_work(self):
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
