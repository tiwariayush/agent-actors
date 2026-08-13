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
spec = importlib.util.spec_from_file_location("json_chain_prompt_ellipsis", module_path)
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


class JSONPromptEllipsisTest(unittest.TestCase):
    def test_parses_plan_array_with_prompt_ellipsis(self):
        """Plan's example ends with `}, ...]` which models commonly copy."""
        output = """\
[
    {
        "task_id": 0,
        "child_id": 0,
        "task": "Look up Sergey Brin's age and multiply by 12",
        "dependencies": []
    },
    ...
]
"""
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Look up Sergey Brin's age and multiply by 12",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_parses_plan_array_with_trailing_comma(self):
        output = """\
[
    {
        "task_id": 0,
        "child_id": 0,
        "task": "Research AGI",
        "dependencies": []
    },
]
"""
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Research AGI",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_parses_adjust_object_with_trailing_comma(self):
        chain = JSONChain(
            '{"confidence": 8, "speak": "Done", "result": "Report ready",}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "Done",
                    "result": "Report ready",
                }
            },
        )

    def test_parses_unicode_ellipsis_placeholder(self):
        chain = JSONChain('[{"task_id": 0, "child_id": 0, "task": "Research", "dependencies": []}, …]')

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

    def test_preserves_ellipsis_inside_task_strings(self):
        chain = JSONChain(
            '[{"task_id": 0, "child_id": 0, "task": "Summarize findings...", "dependencies": []}]'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Summarize findings...",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_preserves_strict_raw_json(self):
        chain = JSONChain('[{"task_id": 0}]')

        self.assertEqual(chain._call({}), {"json": [{"task_id": 0}]})

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
