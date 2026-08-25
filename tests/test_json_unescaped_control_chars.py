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
spec = importlib.util.spec_from_file_location("json_chain_unescaped_controls", module_path)
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


class JSONUnescapedControlCharsTest(unittest.TestCase):
    def test_parses_adjust_multiline_result_string(self):
        """Adjust synthesizes reports; models often emit raw newlines in result."""
        output = """\
{
    "confidence": 8,
    "speak": "The report is ready.",
    "result": "Executive summary:
AGI remains an open research problem.
Recommendation: continue evaluation."
}
"""
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "The report is ready.",
                    "result": (
                        "Executive summary:\n"
                        "AGI remains an open research problem.\n"
                        "Recommendation: continue evaluation."
                    ),
                }
            },
        )

    def test_parent_adjust_path_survives_multiline_result(self):
        """ParentAgent.run reads adjustment['confidence'] after child work."""
        chain = JSONChain(
            '{\n  "confidence": 9,\n  "speak": "Looks good",\n  "result": "Line 1\nLine 2"\n}'
        )
        adjustment = chain._call({})["json"]

        self.assertGreaterEqual(adjustment["confidence"], 8)
        self.assertEqual(adjustment["speak"], "Looks good")
        self.assertEqual(adjustment["result"], "Line 1\nLine 2")

    def test_parses_plan_multiline_task_string(self):
        output = """\
[
    {
        "task_id": 0,
        "child_id": 0,
        "task": "Write the report.
Include five quotes.",
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
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Write the report.\nInclude five quotes.",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_parses_unescaped_tab_and_carriage_return(self):
        chain = JSONChain('{"confidence": 7, "speak": "ok", "result": "a\tb\rc"}')

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 7,
                    "speak": "ok",
                    "result": "a\tb\rc",
                }
            },
        )

    def test_preserves_already_escaped_newlines(self):
        chain = JSONChain(
            '{"confidence": 8, "speak": "Done", "result": "Line 1\\nLine 2"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "Done",
                    "result": "Line 1\nLine 2",
                }
            },
        )

    def test_preserves_newlines_used_as_json_whitespace(self):
        chain = JSONChain(
            '{\n  "confidence": 8,\n  "speak": "Done",\n  "result": "Report ready"\n}'
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

    def test_preserves_strict_raw_json(self):
        chain = JSONChain(
            '[{"task_id": 0, "child_id": 0, "task": "Research", "dependencies": []}]'
        )

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

    def test_rejects_prefix_prose_before_json(self):
        chain = JSONChain('Here is the adjustment: {"confidence": 8}')

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})

    def test_rejects_non_json_content(self):
        chain = JSONChain("not json at all")

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})


if __name__ == "__main__":
    unittest.main()
