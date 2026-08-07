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
spec = importlib.util.spec_from_file_location("json_chain_double_encoded", module_path)
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


class JSONDoubleEncodedTest(unittest.TestCase):
    def test_parses_double_encoded_plan_array(self):
        """Models sometimes JSON-stringify the Plan array as a single string."""
        plan = [
            {
                "task_id": 0,
                "child_id": 0,
                "task": "Research topic",
                "dependencies": [],
            }
        ]
        chain = JSONChain(json.dumps(json.dumps(plan)))

        self.assertEqual(chain._call({}), {"json": plan})

    def test_parses_double_encoded_adjust_object(self):
        adjust = {
            "confidence": 9,
            "speak": "Ready for review",
            "result": "The analysis is complete",
        }
        chain = JSONChain(json.dumps(json.dumps(adjust)))

        self.assertEqual(chain._call({}), {"json": adjust})

    def test_preserves_raw_plan_array(self):
        plan = [{"task_id": 0, "child_id": 0, "task": "Solo", "dependencies": []}]
        chain = JSONChain(json.dumps(plan))

        self.assertEqual(chain._call({}), {"json": plan})

    def test_preserves_raw_adjust_object(self):
        adjust = {"confidence": 8, "speak": "ok", "result": "done"}
        chain = JSONChain(json.dumps(adjust))

        self.assertEqual(chain._call({}), {"json": adjust})

    def test_rejects_non_json_string_payload(self):
        # First parse succeeds as a JSON string; second parse must still be JSON.
        chain = JSONChain(json.dumps("not-json-document"))

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})

    def test_rejects_invalid_content(self):
        chain = JSONChain("Here is the plan: []")

        with self.assertRaises(json.JSONDecodeError):
            chain._call({})


if __name__ == "__main__":
    unittest.main()
