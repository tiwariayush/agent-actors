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


class DependencyFloatRefTest(unittest.TestCase):
    def test_single_dotted_float_in_list(self):
        """JSON `[0.0]` is the unquoted form of the prompt's [worker #.task #]."""
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Multiply by 12",
            dependencies=[0.0],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_nonzero_task_id_float(self):
        record = TaskRecord(
            task_id=2,
            child_id=0,
            task="Write the report",
            dependencies=[0.1],
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 1)

    def test_bare_float_dependency_not_wrapped_in_list(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Multiply by 12",
            dependencies=0.0,
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_multiple_dotted_float_dependencies(self):
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Synthesize findings",
            dependencies=[0.0, 1.2],
        )
        self.assertEqual(
            [dep.id for dep in record.dependencies],
            ["0.0", "1.2"],
        )

    def test_mixed_object_and_float_dependencies(self):
        record = TaskRecord(
            task_id=2,
            child_id=0,
            task="Combine results",
            dependencies=[
                {"child_id": 0, "task_id": 0},
                0.1,
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
            task="Follow-up",
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

    def test_parent_plan_comprehension_accepts_float_dependencies(self):
        plan = [
            {
                "task_id": 0,
                "child_id": 0,
                "task": "Look up Sergey Brin's age",
                "dependencies": [],
            },
            {
                "task_id": 1,
                "child_id": 0,
                "task": "Multiply that age by 12",
                "dependencies": [0.0],
            },
        ]
        records = [TaskRecord(**task) for task in plan]
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_invalid_dependency_shapes_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies=["not-a-ref"],
            )
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies=[0],
            )
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies=[-1.0],
            )


if __name__ == "__main__":
    unittest.main()
