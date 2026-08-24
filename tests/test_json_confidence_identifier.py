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
spec = importlib.util.spec_from_file_location("json_chain_confidence_identifier", module_path)
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


class JSONConfidenceIdentifierTest(unittest.TestCase):
    def test_parses_adjust_confidence_identifier(self):
        """Adjust's example uses unquoted ``confidence`` as the value."""
        output = """\
{
    "confidence": confidence,
    "speak": "The report is ready.",
    "result": "Sergey Brin is 49; 49*12 = 588"
}
"""
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 0,
                    "speak": "The report is ready.",
                    "result": "Sergey Brin is 49; 49*12 = 588",
                }
            },
        )

    def test_parent_adjust_path_survives_confidence_identifier(self):
        """ParentAgent.run compares adjustment['confidence'] after child work."""
        chain = JSONChain(
            '{"confidence": confidence, "speak": "Looks good", "result": "Done"}'
        )
        adjustment = chain._call({})["json"]

        self.assertFalse(adjustment["confidence"] >= 8)
        self.assertEqual(adjustment["speak"], "Looks good")
        self.assertEqual(adjustment["result"], "Done")

    def test_preserves_numeric_confidence(self):
        chain = JSONChain(
            '{"confidence": 9, "speak": "Looks good", "result": "Done"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 9,
                    "speak": "Looks good",
                    "result": "Done",
                }
            },
        )

    def test_preserves_confidence_word_inside_strings(self):
        chain = JSONChain(
            '{"confidence": 8, "speak": "High confidence in this result", "result": "Done"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "High confidence in this result",
                    "result": "Done",
                }
            },
        )

    def test_preserves_quoted_speak_placeholder(self):
        """Adjust quotes ``<what to say to your copilot>``; that is already valid JSON."""
        chain = JSONChain(
            '{"confidence": confidence, "speak": "<what to say to your copilot>", "result": "Done"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 0,
                    "speak": "<what to say to your copilot>",
                    "result": "Done",
                }
            },
        )

    def test_does_not_replace_confidence_prefix_identifiers(self):
        chain = JSONChain(
            '{"confidence": 7, "speak": "ok", "result": "Done", "confidence_score": 7}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 7,
                    "speak": "ok",
                    "result": "Done",
                    "confidence_score": 7,
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
