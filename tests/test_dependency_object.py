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


class TestDependencyObjectNormalization(unittest.TestCase):
    def test_single_dependency_object_is_wrapped(self):
        """Plan LLMs often emit one dependency as an object, not a one-element list."""
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize findings",
            dependencies={"child_id": 0, "task_id": 0},
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")
        self.assertEqual(record.id, "0.1")

    def test_dependency_list_still_works(self):
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Combine reports",
            dependencies=[
                {"child_id": 0, "task_id": 0},
                {"child_id": 0, "task_id": 1},
            ],
        )
        self.assertEqual([dep.id for dep in record.dependencies], ["0.0", "0.1"])

    def test_empty_dependencies_preserved(self):
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Solo research",
            dependencies=[],
        )
        self.assertEqual(record.dependencies, [])

    def test_missing_dependencies_default_to_empty_list(self):
        record = TaskRecord(task_id=0, child_id=0, task="No deps key")
        self.assertEqual(record.dependencies, [])

    def test_parent_plan_comprehension_accepts_object_dependency(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "task_id": 0,
                "child_id": 0,
                "task": "Research AGI",
                "dependencies": [],
            },
            {
                "task_id": 1,
                "child_id": 0,
                "task": "Write summary",
                "dependencies": {"child_id": 0, "task_id": 0},
            },
        ]
        planned_tasks = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(planned_tasks[1].dependencies[0].id, "0.0")

    def test_invalid_dependencies_type_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies="0.0",
            )


if __name__ == "__main__":
    unittest.main()
