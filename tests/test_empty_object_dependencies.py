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


class TestEmptyObjectDependencies(unittest.TestCase):
    def test_empty_object_dependencies_become_empty_list(self):
        """Plan LLMs often emit {} for tasks with no dependencies."""
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies={},
        )
        self.assertEqual(record.dependencies, [])
        self.assertEqual(record.id, "0.0")

    def test_parent_plan_comprehension_accepts_empty_object_dependencies(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "task_id": 0,
                "child_id": 0,
                "task": "Solo research",
                "dependencies": {},
            },
            {
                "task_id": 1,
                "child_id": 0,
                "task": "Write summary",
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        planned_tasks = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(planned_tasks[0].dependencies, [])
        self.assertEqual(planned_tasks[1].dependencies[0].id, "0.0")

    def test_missing_dependencies_default_to_empty_list(self):
        record = TaskRecord(task_id=1, child_id=2, task="Write summary")
        self.assertEqual(record.dependencies, [])

    def test_explicit_empty_list_dependencies_preserved(self):
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Solo task",
            dependencies=[],
        )
        self.assertEqual(record.dependencies, [])

    def test_populated_dependencies_preserved(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize findings",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_non_empty_dependency_object_still_rejected_here(self):
        """Bare non-empty objects remain draft PR #75's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies={"child_id": 0, "task_id": 0},
            )

    def test_top_level_null_still_rejected_here(self):
        """Top-level null remains draft PR #71's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Research AGI",
                dependencies=None,
            )

    def test_null_placeholder_in_array_still_rejected_here(self):
        """Null placeholders inside the array remain draft PR #79's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Research AGI",
                dependencies=[None],
            )


if __name__ == "__main__":
    unittest.main()
