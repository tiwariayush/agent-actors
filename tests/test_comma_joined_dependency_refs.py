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


class TestCommaJoinedDependencyRefs(unittest.TestCase):
    def test_bare_comma_joined_dotted_refs(self):
        """Plan prompt citations get concatenated: dependencies: \"0.0, 1.0\"."""
        record = TaskRecord(
            task_id=2,
            child_id=0,
            task="Write the executive report",
            dependencies="0.0, 1.0",
        )
        self.assertEqual([dep.id for dep in record.dependencies], ["0.0", "1.0"])

    def test_bracket_comma_joined_dotted_refs(self):
        record = TaskRecord(
            task_id=2,
            child_id=0,
            task="Write the executive report",
            dependencies="[0.0], [1.0]",
        )
        self.assertEqual([dep.id for dep in record.dependencies], ["0.0", "1.0"])

    def test_single_bracket_with_multiple_dotted_refs(self):
        record = TaskRecord(
            task_id=3,
            child_id=1,
            task="Combine jokes and quotes",
            dependencies="[0.0, 1.0, 2.0]",
        )
        self.assertEqual(
            [dep.id for dep in record.dependencies],
            ["0.0", "1.0", "2.0"],
        )

    def test_no_spaces_between_refs(self):
        record = TaskRecord(
            task_id=2,
            child_id=0,
            task="Synthesize findings",
            dependencies="0.0,1.0",
        )
        self.assertEqual([dep.id for dep in record.dependencies], ["0.0", "1.0"])

    def test_nonzero_child_and_task_ids(self):
        record = TaskRecord(
            task_id=1,
            child_id=42,
            task="Assemble jokes and quotes",
            dependencies="2.0, 42.1",
        )
        self.assertEqual(record.dependencies[0].child_id, 2)
        self.assertEqual(record.dependencies[0].task_id, 0)
        self.assertEqual(record.dependencies[1].child_id, 42)
        self.assertEqual(record.dependencies[1].task_id, 1)

    def test_list_entry_with_comma_joined_refs(self):
        record = TaskRecord(
            task_id=2,
            child_id=0,
            task="Write the executive report",
            dependencies=["0.0, 1.0"],
        )
        self.assertEqual([dep.id for dep in record.dependencies], ["0.0", "1.0"])

    def test_mixed_object_and_comma_joined_string(self):
        record = TaskRecord(
            task_id=3,
            child_id=0,
            task="Combine reports",
            dependencies=[
                {"child_id": 0, "task_id": 0},
                "1.0, 2.0",
            ],
        )
        self.assertEqual(
            [dep.id for dep in record.dependencies],
            ["0.0", "1.0", "2.0"],
        )

    def test_parent_plan_comprehension_accepts_comma_joined_refs(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "task_id": 0,
                "child_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "task_id": 0,
                "child_id": 1,
                "task": "Collect relevant jokes",
                "dependencies": [],
            },
            {
                "task_id": 0,
                "child_id": 2,
                "task": "Write the executive report",
                "dependencies": "0.0, 1.0",
            },
        ]
        planned_tasks = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(
            [dep.id for dep in planned_tasks[2].dependencies],
            ["0.0", "1.0"],
        )

    def test_object_list_dependencies_still_work(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Follow-up",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_empty_and_missing_dependencies_preserved(self):
        self.assertEqual(
            TaskRecord(task_id=0, child_id=0, task="Solo").dependencies,
            [],
        )
        self.assertEqual(
            TaskRecord(
                task_id=0, child_id=0, task="Solo", dependencies=[]
            ).dependencies,
            [],
        )

    def test_single_string_ref_still_rejected_here(self):
        """Quoted '0.0' / '[0.0]' remain draft PR #76's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies="0.0",
            )
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies="[0.0]",
            )

    def test_integer_pair_string_still_rejected_here(self):
        """'[0, 0]' remains draft PR #77's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies="[0, 0]",
            )

    def test_stringified_empty_array_still_rejected_here(self):
        """Stringified '[]' remains draft PR #90's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Research AGI",
                dependencies="[]",
            )

    def test_prose_around_refs_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies="depends on 0.0, 1.0",
            )


if __name__ == "__main__":
    unittest.main()
