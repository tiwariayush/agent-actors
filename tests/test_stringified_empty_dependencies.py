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


class TestStringifiedEmptyDependencies(unittest.TestCase):
    def test_stringified_empty_array_becomes_empty_list(self):
        """Models often JSON-encode an empty dependency list as the string '[]'."""
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies="[]",
        )
        self.assertEqual(record.dependencies, [])
        self.assertEqual(record.id, "0.0")

    def test_whitespace_padded_stringified_empty_array(self):
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies=" [ ] ",
        )
        self.assertEqual(record.dependencies, [])

    def test_empty_string_dependencies_become_empty_list(self):
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Solo task",
            dependencies="",
        )
        self.assertEqual(record.dependencies, [])

    def test_stringified_null_becomes_empty_list(self):
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies="null",
        )
        self.assertEqual(record.dependencies, [])

    def test_stringified_empty_object_becomes_empty_list(self):
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies="{}",
        )
        self.assertEqual(record.dependencies, [])

    def test_parent_plan_comprehension_accepts_stringified_empty_deps(self):
        """ParentAgent.run builds TaskRecords via TaskRecord(**t) over plan JSON."""
        plan_json = [
            {
                "task_id": 0,
                "child_id": 0,
                "task": "Look up Sergey Brin's age",
                "dependencies": "[]",
            },
            {
                "task_id": 1,
                "child_id": 0,
                "task": "Multiply that age by 12",
                "dependencies": [{"child_id": 0, "task_id": 0}],
            },
        ]
        planned_tasks = [TaskRecord(**t) for t in plan_json]
        self.assertEqual(planned_tasks[0].dependencies, [])
        self.assertEqual(planned_tasks[1].dependencies[0].id, "0.0")

    def test_missing_and_empty_list_dependencies_preserved(self):
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

    def test_populated_dependencies_preserved(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Synthesize findings",
            dependencies=[{"child_id": 0, "task_id": 0}],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_string_task_ref_still_rejected_here(self):
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

    def test_top_level_null_still_rejected_here(self):
        """JSON null remains draft PR #71's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Research AGI",
                dependencies=None,
            )

    def test_empty_object_still_rejected_here(self):
        """Bare {} remains draft PR #83's responsibility."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Research AGI",
                dependencies={},
            )


if __name__ == "__main__":
    unittest.main()
