import unittest

from pydantic import ValidationError

from agent_actors.models import TaskRecord


class TestTaskRecordDependencies(unittest.TestCase):
    def test_null_dependencies_become_empty_list(self):
        """Plan LLMs often emit JSON null for tasks with no dependencies."""
        record = TaskRecord(
            task_id=0,
            child_id=0,
            task="Research AGI",
            dependencies=None,
        )
        self.assertEqual(record.dependencies, [])
        self.assertEqual(record.id, "0.0")

    def test_missing_dependencies_default_to_empty_list(self):
        record = TaskRecord(task_id=1, child_id=2, task="Write summary")
        self.assertEqual(record.dependencies, [])

    def test_explicit_empty_dependencies_preserved(self):
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

    def test_invalid_dependencies_type_still_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies="none",
            )


if __name__ == "__main__":
    unittest.main()
