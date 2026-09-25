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


class TestWorkerTaskCitations(unittest.TestCase):
    def test_prompt_literal_citation_in_dependency_list(self):
        """Plan prompt: When referencing other tasks, use [worker #.task #]."""
        record = TaskRecord(
            task_id=0,
            child_id=1,
            task="Write the report",
            dependencies=["[worker 0.task 0]"],
        )
        self.assertEqual(len(record.dependencies), 1)
        self.assertEqual(record.dependencies[0].id, "0.0")

    def test_citation_string_not_wrapped_in_list(self):
        record = TaskRecord(
            task_id=0,
            child_id=1,
            task="Write the report",
            dependencies="[worker 0.task 0]",
        )
        self.assertEqual(record.dependencies[0].child_id, 0)
        self.assertEqual(record.dependencies[0].task_id, 0)

    def test_comma_joined_prompt_citations(self):
        record = TaskRecord(
            task_id=1,
            child_id=2,
            task="Combine reports",
            dependencies="[worker 0.task 0], [worker 1.task 0]",
        )
        self.assertEqual(
            [dep.id for dep in record.dependencies],
            ["0.0", "1.0"],
        )

    def test_comma_joined_citation_inside_list(self):
        record = TaskRecord(
            task_id=1,
            child_id=2,
            task="Combine reports",
            dependencies=["[worker 0.task 0], [worker 42.task 1]"],
        )
        self.assertEqual(
            [(dep.child_id, dep.task_id) for dep in record.dependencies],
            [(0, 0), (42, 1)],
        )

    def test_hash_marker_and_whitespace_and_case(self):
        record = TaskRecord(
            task_id=1,
            child_id=0,
            task="Draft",
            dependencies=["[ Worker #42 . Task #3 ]"],
        )
        self.assertEqual(record.dependencies[0].id, "42.3")

    def test_mixed_object_and_prompt_citation(self):
        record = TaskRecord(
            task_id=2,
            child_id=1,
            task="Combine reports",
            dependencies=[
                {"child_id": 0, "task_id": 0},
                "[worker 1.task 0]",
            ],
        )
        self.assertEqual(
            [dep.id for dep in record.dependencies],
            ["0.0", "1.0"],
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

    def test_json_loads_then_task_record(self):
        raw = """
        [
          {
            "task_id": 0,
            "child_id": 0,
            "task": "Research AGI",
            "dependencies": []
          },
          {
            "task_id": 0,
            "child_id": 1,
            "task": "Write the report",
            "dependencies": ["[worker 0.task 0]"]
          }
        ]
        """
        records = [TaskRecord(**task) for task in json.loads(raw)]
        self.assertEqual(records[1].dependencies[0].id, "0.0")
        self.assertEqual(str(records[1]).split(" (depends on ")[1], "[0.0])")

    def test_parent_plan_comprehension_accepts_prompt_citations(self):
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
                "task": "Write the report",
                "dependencies": ["[worker 0.task 0]"],
            },
        ]
        records = [TaskRecord(**task) for task in plan_json]
        self.assertEqual(records[1].dependencies[0].id, "0.0")

    def test_dotted_bracket_refs_stay_rejected(self):
        """"[0.0]" is a different citation shape and must not be parsed here."""
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Write summary",
                dependencies=["[0.0]"],
            )

    def test_bare_dotted_refs_stay_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Write summary",
                dependencies=["0.0"],
            )

    def test_integer_pair_refs_stay_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Write summary",
                dependencies=[[0, 0]],
            )

    def test_comma_joined_dotted_refs_stay_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=1,
                child_id=0,
                task="Write summary",
                dependencies="0.0, 1.0",
            )

    def test_null_dependencies_stay_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=None,
            )

    def test_prose_around_citation_stays_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=["see [worker 0.task 0]"],
            )

    def test_incomplete_citation_stays_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=["[worker 0.task]"],
            )

    def test_unrelated_string_stays_rejected(self):
        with self.assertRaises(ValidationError):
            TaskRecord(
                task_id=0,
                child_id=0,
                task="Broken",
                dependencies=["not-a-ref"],
            )


if __name__ == "__main__":
    unittest.main()
