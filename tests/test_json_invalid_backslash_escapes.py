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
spec = importlib.util.spec_from_file_location("json_chain_invalid_backslash", module_path)
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


class JSONInvalidBackslashEscapesTest(unittest.TestCase):
    def test_parses_adjust_result_with_readme_url_citation(self):
        """README citation uses LaTeX \\url; models copy it into Adjust result."""
        output = (
            '{"confidence": 8, "speak": "The post is ready.", '
            '"result": "howpublished = {\\url{https://github.com/shaman-ai/agent-actors}}"}'
        )
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "The post is ready.",
                    "result": (
                        "howpublished = "
                        "{\\url{https://github.com/shaman-ai/agent-actors}}"
                    ),
                }
            },
        )

    def test_parent_adjust_path_survives_invalid_backslash(self):
        """ParentAgent.run reads adjustment['confidence'] after child work."""
        chain = JSONChain(
            '{"confidence": 9, "speak": "Looks good", '
            '"result": "Cite as \\url{https://example.com}"}'
        )
        adjustment = chain._call({})["json"]

        self.assertGreaterEqual(adjustment["confidence"], 8)
        self.assertEqual(adjustment["speak"], "Looks good")
        self.assertEqual(adjustment["result"], "Cite as \\url{https://example.com}")

    def test_parses_regex_invalid_escape_in_result(self):
        chain = JSONChain(
            '{"confidence": 7, "speak": "ok", "result": "Pattern: \\d+ matches digits"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 7,
                    "speak": "ok",
                    "result": "Pattern: \\d+ matches digits",
                }
            },
        )

    def test_parses_windows_path_invalid_unicode_escape(self):
        # Use \\out rather than \\report so a legal JSON \\r escape is not
        # mistaken for the start of "report".
        chain = JSONChain(
            '{"confidence": 8, "speak": "saved", "result": "Wrote C:\\Users\\agent\\out.txt"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "saved",
                    "result": "Wrote C:\\Users\\agent\\out.txt",
                }
            },
        )

    def test_parses_plan_task_with_latex_cite(self):
        output = (
            '[{"task_id": 0, "child_id": 0, '
            '"task": "Summarize \\cite{agentactors}.", "dependencies": []}]'
        )
        chain = JSONChain(output)

        self.assertEqual(
            chain._call({}),
            {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Summarize \\cite{agentactors}.",
                        "dependencies": [],
                    }
                ]
            },
        )

    def test_preserves_already_escaped_backslashes(self):
        chain = JSONChain(
            '{"confidence": 8, "speak": "Done", "result": "Literal \\\\d+ and \\\\url"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "Done",
                    "result": "Literal \\d+ and \\url",
                }
            },
        )

    def test_preserves_valid_unicode_and_tab_escapes(self):
        chain = JSONChain(
            '{"confidence": 8, "speak": "ok", "result": "A\\u0041\\tB"}'
        )

        self.assertEqual(
            chain._call({}),
            {
                "json": {
                    "confidence": 8,
                    "speak": "ok",
                    "result": "AA\tB",
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
