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


class TestDependencyPairRefs(unittest.TestCase):
    def test_nested_integer_pair_dependency(self):
        """Prose format [worker #.task #] often becomes JSON [[worker, task]]."""
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Summarize research",
            dependencies=[[0, 0]],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_flat_integer_pair_dependencies_value(self):
        """A single dependency may be emitted as [worker, task] at the top level."""
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Summarize research",
            dependencies=[0, 0],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)

    def test_multiple_nested_integer_pairs(self):
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Combine reports",
            dependencies=[[0, 0], [0, 1]],
        )
        self.assertEqual(
            [dep.id for dep in record.dependencies],
            ["0.0", "0.1"],
        )

    def test_comma_bracket_string_dependency(self):
        """Models may stringify the prose form with a comma: '[0, 0]'."""
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Summarize research",
            dependencies=["[0, 0]"],
        )
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_single_comma_bracket_string_not_wrapped_in_list(self):
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Combine reports",
            dependencies="[0, 1]",
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.1")

    def test_mixed_object_and_pair_dependencies(self):
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Combine reports",
            dependencies=[
                {"child_id": 0, "task_id": 0},
                [0, 1],
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

    def test_parent_plan_comprehension_accepts_pair_dependencies(self):
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
                "dependencies": [[0, 0]],
            },
        ]
        records = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_invalid_dependency_shapes_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=["not-a-ref"],
            )
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=[[0]],
            )
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=[0, 0, 1],
            )


if __name__ == "__main__":
    unittest.main()
