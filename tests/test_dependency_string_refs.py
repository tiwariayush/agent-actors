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


class TestDependencyStringRefs(unittest.TestCase):
    def test_bracket_string_dependency_in_list(self):
        """Plan prompt tells models to reference tasks as [worker #.task #]."""
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Summarize research",
            dependencies=["[0.0]"],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_bare_string_dependency_in_list(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Summarize research",
            dependencies=["0.0"],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)

    def test_single_string_dependency_not_wrapped_in_list(self):
        """Models sometimes emit one dependency as a bare string, not a list."""
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Combine reports",
            dependencies="[0.1]",
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.1")

    def test_mixed_object_and_string_dependencies(self):
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Combine reports",
            dependencies=[
                {"child_id": 0, "task_id": 0},
                "[0.1]",
            ],
        )
        self.assertEqual(
            [dep.id for dep in record.dependencies],
            ["0.0", "0.1"],
        )

    def test_object_list_dependencies_still_work(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize findings",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_empty_and_missing_dependencies_preserved(self):
        self.assertEqual(
            TaskRecord(
                task_id=0, child_id=0, task="Solo", dependencies=[]
            ).dependencies,
            [],
        )
        self.assertEqual(
            TaskRecord(task_id=0, child_id=0, task="Solo").dependencies,
            [],
        )

    def test_parent_plan_comprehension_accepts_string_dependencies(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "task_id": 0,
                "child_id": 0,
                "task": "Research topic",
                "dependencies": [],
            },
            {
                "task_id": 1,
                "child_id": 0,
                "task": "Write summary",
                "dependencies": ["[0.0]"],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_invalid_dependency_string_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=["not-a-ref"],
            )


if __name__ == "__main__":
    unittest.main()
