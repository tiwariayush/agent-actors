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
spec = importlib.util.spec_from_file_location("json_chain_trailing_prose", module_path)
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


class JSONTrailingProseTest(unittest.TestCase):
    def test_parses_plan_array_with_trailing_prose(self):
        output = (
            '[{"task_id": 0, "child_id": 0, "task": "Research topic", '
            '"dependencies": []}]\n'
            "I assigned this to the best-suited team member."
        )
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Research topic",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_parses_adjust_object_with_trailing_prose(self):
        output = (
            '{"confidence": 8, "speak": "Done", "result": "Report ready"}\n'
            "Hope this completes the objective."
        )
        chain = JSONChain(output)

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
        chain = JSONChain('[{"task_id": 0}]')

        self.assertEqual(chain._call({}), {"json": [{"task_id": 0}]})

    def test_rejects_prefix_prose_before_json(self):
        chain = JSONChain('Here is the result: {"confidence": 8}')

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})

    def test_rejects_non_json_content(self):
        chain = JSONChain("not json at all")

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})


if __name__ == "__main__":
    unittest.main()
