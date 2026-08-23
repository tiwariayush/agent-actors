import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


class StubLLMChain:
    def __init__(self, output):
        self.output = output

    def _call(self, inputs):
        return {"json": self.output}


langchain = types.ModuleType("langchain")
langchain.LLMChain = StubLLMChain

module_path = Path(__file__).parents[1] / "agent_actors" / "chains" / "base.py"
spec = importlib.util.spec_from_file_location("json_chain_angle_placeholders", module_path)
json_chain_module = importlib.util.module_from_spec(spec)
previous_langchain = sys.modules.get("langchain")
try:
    sys.modules["langchain"] = langchain
    spec.loader.exec_module(json_chain_module)
finally:
    if previous_langchain is None:
        del sys.modules["langchain"]
    else:
        sys.modules["langchain"] = previous_langchain
JSONChain = json_chain_module.JSONChain


class JSONAnglePlaceholderTest(unittest.TestCase):
    def test_parses_adjust_result_placeholder(self):
        """Adjust's example uses unquoted `<synthesized result...>` for result."""
        output = """\
{
    "confidence": 8,
    "speak": "The report is ready.",
    "result": <synthesized result satisfying objective>
}
"""
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "The report is ready.",
                    "result": None,
                }
            },
        )

    def test_parses_plan_unquoted_id_placeholders(self):
        """Plan's example uses unquoted `<incrementing int...>` / `<int>`."""
        output = """\
[
    {
        "task_id": <incrementing int starting at 0 per child>,
        "child_id": <assigned team member id>,
        "task": "Look up Sergey Brin's age",
        "dependencies": []
    }
]
"""
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": None,
                        "child_id": None,
                        "task": "Look up Sergey Brin's age",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_parses_plan_dependency_int_placeholder(self):
        output = """\
[
    {
        "task_id": 1,
        "child_id": 0,
        "task": "Multiply that age by 12",
        "dependencies": [{
            "child_id": 0,
            "task_id": <int>
        }]
    }
]
"""
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 1,
                        "child_id": 0,
                        "task": "Multiply that age by 12",
                        "dependencies": [{"child_id": 0, "task_id": None}],
                    }
                ]
            },
        )

    def test_preserves_quoted_speak_placeholder(self):
        """Adjust quotes `<what to say to your copilot>`; that is already valid JSON."""
        chain = JSONChain(
            '{"confidence": 9, "speak": "<what to say to your copilot>", "result": "Done"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 9,
                    "speak": "<what to say to your copilot>",
                    "result": "Done",
                }
            },
        )

    def test_preserves_angle_brackets_inside_task_strings(self):
        chain = JSONChain(
            '[{"task_id": 0, "child_id": 0, "task": "Parse <int> from the input", "dependencies": []}]'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Parse <int> from the input",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_preserves_strict_raw_json(self):
        chain = JSONChain('[{"task_id": 0, "child_id": 0, "task": "Research", "dependencies": []}]')

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Research",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_parent_adjust_path_survives_result_placeholder(self):
        """ParentAgent.run reads adjustment['confidence'] after child work finishes."""
        chain = JSONChain(
            '{"confidence": 8, "speak": "Looks good", "result": <synthesized result satisfying objective>}'
        )
        adjustment = chain._call({})["json"]

        self.assertGreaterEqual(adjustment["confidence"], 8)
        self.assertEqual(adjustment["speak"], "Looks good")
        self.assertIsNone(adjustment["result"])

    def test_rejects_prefix_prose_before_json(self):
        chain = JSONChain('Here is the plan: [{"task_id": 0}]')

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})

    def test_rejects_non_json_content(self):
        chain = JSONChain("not json at all")

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})


if __name__ == "__main__":
    unittest.main()
