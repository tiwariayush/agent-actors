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


class TestNullDependencyPlaceholders(unittest.TestCase):
    def test_null_item_in_dependencies_becomes_empty_list(self):
        """Prompt-shaped array with a null placeholder means no dependencies."""
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies=[None],
        )
        self.assertEqual(record.dependencies, [])
        self.assertEqual(record.id, "0.0")

    def test_null_id_object_placeholder_dropped(self):
        """Models copy the prompt's dependency object and null both ids."""
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies=[{"child_id": None, "task_id": None}],
        )
        self.assertEqual(record.dependencies, [])

    def test_empty_object_placeholder_dropped(self):
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Solo task",
            dependencies=[{}],
        )
        self.assertEqual(record.dependencies, [])

    def test_mixed_real_dependency_and_null_placeholder(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize findings",
            dependencies=[
                {"child_id": 0, "task_id": 0},
                None,
                {"child_id": None, "task_id": None},
            ],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_populated_dependencies_preserved(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Write summary",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)

    def test_partial_null_dependency_still_rejected(self):
        """A half-specified dependency is incomplete, not an empty placeholder."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Broken",
                dependencies=[{"child_id": 0, "task_id": None}],
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


if __name__ == "__main__":
    unittest.main()
