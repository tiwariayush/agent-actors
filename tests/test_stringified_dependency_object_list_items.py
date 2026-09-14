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
_stringified_dependency_object_list_items = (
    models._stringified_dependency_object_list_items
)


class StringifiedDependencyObjectListItemHelperTests(unittest.TestCase):
    def test_list_of_one_stringified_object(self):
        self.assertEqual(
            _stringified_dependency_object_list_items(
                ['{"child_id": 0, "task_id": 0}']
            ),
            [{"child_id": 0, "task_id": 0}],
        )

    def test_whitespace_padded_stringified_object_item(self):
        self.assertEqual(
            _stringified_dependency_object_list_items(
                [' { "child_id": 0, "task_id": 0 } ']
            ),
            [{"child_id": 0, "task_id": 0}],
        )

    def test_list_of_multiple_stringified_objects(self):
        self.assertEqual(
            _stringified_dependency_object_list_items(
                [
                    '{"child_id": 0, "task_id": 0}',
                    '{"child_id": 1, "task_id": 0}',
                ]
            ),
            [
                {"child_id": 0, "task_id": 0},
                {"child_id": 1, "task_id": 0},
            ],
        )

    def test_mixed_stringified_and_parsed_objects(self):
        parsed = {"child_id": 1, "task_id": 0}
        self.assertEqual(
            _stringified_dependency_object_list_items(
                ['{"child_id": 0, "task_id": 0}', parsed]
            ),
            [{"child_id": 0, "task_id": 0}, parsed],
        )

    def test_stringified_object_with_extra_keys(self):
        self.assertEqual(
            _stringified_dependency_object_list_items(
                ['{"child_id": 1, "task_id": 2, "note": "prior"}']
            ),
            [{"child_id": 1, "task_id": 2, "note": "prior"}],
        )

    def test_canonical_list_left_unchanged(self):
        raw = [{"child_id": 0, "task_id": 0}]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_empty_list_left_unchanged(self):
        raw = []
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_bare_object_left_unchanged(self):
        """A parsed dict stays with the existing bare-object wrap."""
        raw = {"child_id": 0, "task_id": 0}
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_stringified_object_field_left_unchanged(self):
        """A stringified bare object stays with that repair."""
        raw = '{"child_id": 0, "task_id": 0}'
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_stringified_object_array_field_left_unchanged(self):
        """A stringified object array stays with that repair."""
        raw = '[{"child_id": 0, "task_id": 0}]'
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_citation_string_items_left_unchanged(self):
        raw = ["0.0"]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_bracketed_citation_string_items_left_unchanged(self):
        raw = ["[0.0]"]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_empty_object_string_item_left_unchanged(self):
        raw = ["{}"]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_empty_array_string_item_left_unchanged(self):
        raw = ["[]"]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_null_string_item_left_unchanged(self):
        raw = ["null"]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_integer_pair_left_unchanged(self):
        raw = [0, 0]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_integer_pair_nested_left_unchanged(self):
        raw = [[0, 0]]
        self.assertIs(_stringified_dependency_object_list_items(raw), raw)

    def test_null_left_unchanged(self):
        self.assertIsNone(_stringified_dependency_object_list_items(None))


class StringifiedDependencyObjectListItemRecordTests(unittest.TestCase):
    def test_list_of_one_stringified_object_constructs(self):
        record = TaskRecord(
            child_id=0,
            task_id=1,
            task="Write the executive report",
            dependencies=['{"child_id": 0, "task_id": 0}'],
        )
        self.assertEqual(record.child_id, 0)
        self.assertEqual(record.task_id, 1)
        self.assertEqual(record.task, "Write the executive report")
        self.assertEqual(record.id, "0.1")
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_json_loads_list_of_stringified_objects(self):
        """Plan JSON that stringifies dependency objects as list items is valid JSON then crashed."""
        payload = json.loads(
            '{"child_id": 0, "task_id": 1,'
            ' "task": "Write the executive report",'
            ' "dependencies": ["{\\"child_id\\": 0, \\"task_id\\": 0}"]}'
        )
        self.assertIsInstance(payload["dependencies"], list)
        self.assertIsInstance(payload["dependencies"][0], str)
        record = TaskRecord(**payload)
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.id, "0.1")

    def test_json_loads_multiple_stringified_object_items(self):
        payload = json.loads(
            '{"child_id": 2, "task_id": 0,'
            ' "task": "Synthesize findings",'
            ' "dependencies": ['
            ' "{\\"child_id\\": 0, \\"task_id\\": 0}",'
            ' "{\\"child_id\\": 1, \\"task_id\\": 0}"'
            "]}"
        )
        record = TaskRecord(**payload)
        self.assertEqual([d.id for d in record.dependencies], ["0.0", "1.0"])
        self.assertEqual(record.id, "2.0")

    def test_whitespace_padded_stringified_object_item_constructs(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Synthesize findings",
            dependencies=[' { "child_id": 1, "task_id": 0 } '],
        )
        self.assertEqual(record.dependencies[0].id, "1.0")
        self.assertEqual(record.id, "2.0")

    def test_mixed_stringified_and_parsed_objects_construct(self):
        record = TaskRecord(
            child_id=2,
            task_id=0,
            task="Write the executive report",
            dependencies=[
                '{"child_id": 0, "task_id": 0}',
                {"child_id": 1, "task_id": 0},
            ],
        )
        self.assertEqual([d.id for d in record.dependencies], ["0.0", "1.0"])

    def test_canonical_list_still_works(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.task, "Synthesize")
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_parent_plan_comprehension_accepts_stringified_object_items(self):
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
                "dependencies": ['{"child_id": 0, "task_id": 0}'],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[0].id, "0.0")
        self.assertEqual(records[1].id, "2.0")
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_stringified_object_field_still_rejected(self):
        """A stringified bare object remains a separate unwrap."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies='{"child_id": 0, "task_id": 0}',
            )

    def test_stringified_object_array_field_still_rejected(self):
        """A stringified object array remains a separate array unwrap."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies='[{"child_id": 0, "task_id": 0}]',
            )

    def test_citation_string_items_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies=["0.0"],
            )

    def test_bracketed_citation_string_items_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies=["[0.0]"],
            )

    def test_empty_object_string_item_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=0,
                task="Research AGI",
                dependencies=["{}"],
            )

    def test_empty_array_string_item_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=0,
                task="Research AGI",
                dependencies=["[]"],
            )

    def test_integer_pair_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                child_id=0,
                task_id=1,
                task="Write the report",
                dependencies=[0, 0],
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
                    "dependencies": ['{"child_id": 0, "task_id": 0}'],
                }
            )


class StringifiedDependencyObjectListItemRefTests(unittest.TestCase):
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
