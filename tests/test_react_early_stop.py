"""Early-stop must keep the final reply when it is not valid ReAct."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_child_module():
    def _install(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    for name in [
        "langchain",
        "langchain.agents",
        "langchain.chat_models",
        "langchain.chat_models.base",
        "langchain.schema",
    ]:
        _install(name)

    class AgentFinish:
        def __init__(self, return_values, log):
            self.return_values = return_values
            self.log = log

    class OutputParserException(Exception):
        pass

    class ZeroShotAgent:
        pass

    class MRKLChain:
        pass

    class LLMChain:
        pass

    class PromptTemplate:
        pass

    class Tool:
        pass

    class BaseChatModel:
        pass

    sys.modules["langchain"].LLMChain = LLMChain
    sys.modules["langchain"].MRKLChain = MRKLChain
    sys.modules["langchain"].PromptTemplate = PromptTemplate
    sys.modules["langchain.agents"].Tool = Tool
    sys.modules["langchain.agents"].ZeroShotAgent = ZeroShotAgent
    sys.modules["langchain.chat_models.base"].BaseChatModel = BaseChatModel
    sys.modules["langchain.schema"].AgentFinish = AgentFinish
    sys.modules["langchain.schema"].OutputParserException = OutputParserException

    spec = importlib.util.spec_from_file_location(
        "agent_actors_child_under_test",
        Path(__file__).resolve().parents[1] / "agent_actors" / "chains" / "child.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.AgentFinish = AgentFinish
    module.OutputParserException = OutputParserException
    return module


class _Action:
    def __init__(self, log):
        self.log = log


class _Chain:
    def __init__(self, reply):
        self.reply = reply
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return self.reply


class _Parser:
    def __init__(self, module):
        self.module = module

    def parse(self, text):
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            return self.module.AgentFinish({"output": answer}, text)
        if "Action:" in text and "Action Input:" in text:
            return ("tool", text)
        raise self.module.OutputParserException(
            f"Could not parse LLM output: `{text}`"
        )


class ReActEarlyStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.child = _load_child_module()

    def _agent(self, reply):
        agent = self.child.ReActAgent()
        agent.llm_chain = _Chain(reply)
        agent.output_parser = _Parser(self.child)
        agent.observation_prefix = "Observation: "
        agent.llm_prefix = "Thought:"
        agent._stop = ["\nObservation:"]
        return agent

    def test_unparseable_final_reply_is_returned(self):
        agent = self._agent("Sergey Brin is 49, times 12 is 588.")
        stopped = agent.return_stopped_response(
            "generate",
            [(_Action("Thought: look it up"), "49 years old")],
            context="ctx",
            task="age times 12",
        )

        self.assertIsInstance(stopped, self.child.AgentFinish)
        self.assertEqual(
            stopped.return_values["output"],
            "Sergey Brin is 49, times 12 is 588.",
        )
        self.assertIn("49 years old", agent.llm_chain.kwargs["agent_scratchpad"])
        self.assertEqual(agent.llm_chain.kwargs["task"], "age times 12")
        self.assertEqual(agent.llm_chain.kwargs["stop"], ["\nObservation:"])

    def test_final_answer_prefix_is_still_parsed(self):
        agent = self._agent("Final Answer: 588")
        stopped = agent.return_stopped_response("generate", [], task="age times 12")

        self.assertEqual(stopped.return_values["output"], "588")

    def test_parsed_tool_action_falls_back_to_raw_text(self):
        reply = "Action: Search\nAction Input: sergey brin age"
        agent = self._agent(reply)
        stopped = agent.return_stopped_response("generate", [])

        self.assertEqual(stopped.return_values["output"], reply)

    def test_llm_failures_still_propagate(self):
        agent = self._agent("unused")

        def _explode(**kwargs):
            raise RuntimeError("llm down")

        agent.llm_chain.predict = _explode
        with self.assertRaises(RuntimeError):
            agent.return_stopped_response("generate", [])


if __name__ == "__main__":
    unittest.main()
