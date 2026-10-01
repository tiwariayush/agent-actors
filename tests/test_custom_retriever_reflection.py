"""Reflection must not crash when the retriever has no memory_stream."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_agent_module():
    class Tool:
        pass

    class BaseChatModel:
        pass

    class BaseRetriever:
        def get_relevant_documents(self, query):
            return []

        async def aget_relevant_documents(self, query):
            return []

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class LLMChain:
        def run(self, **kwargs):
            raise AssertionError(f"unexpected chain call: {kwargs}")

    class ObjectRef:
        pass

    class AgentActor:
        @staticmethod
        def remote(agent):
            return object()

    def _install(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    for name in [
        "ray",
        "langchain",
        "langchain.schema",
        "langchain.agents",
        "langchain.chat_models",
        "langchain.chat_models.base",
        "agent_actors",
        "agent_actors.actors",
        "agent_actors.chains",
        "agent_actors.chains.agent",
    ]:
        _install(name)

    sys.modules["ray"].ObjectRef = ObjectRef
    sys.modules["langchain"].LLMChain = LLMChain
    sys.modules["langchain.schema"].BaseRetriever = BaseRetriever
    sys.modules["langchain.schema"].Document = Document
    sys.modules["langchain.agents"].Tool = Tool
    sys.modules["langchain.chat_models.base"].BaseChatModel = BaseChatModel
    sys.modules["agent_actors.actors"].AgentActor = AgentActor

    class _Chain:
        @classmethod
        def from_llm(cls, **kwargs):
            return LLMChain()

    chains = sys.modules["agent_actors.chains.agent"]
    chains.WorkingMemory = type("WorkingMemory", (_Chain,), {})
    chains.Synthesis = type("Synthesis", (_Chain,), {})
    chains.GenerateInsights = type("GenerateInsights", (_Chain,), {})
    chains.MemoryStrength = type("MemoryStrength", (_Chain,), {})

    spec = importlib.util.spec_from_file_location(
        "agent_actors_agent_under_test",
        Path(__file__).resolve().parents[1] / "agent_actors" / "agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BaseRetriever = BaseRetriever
    module.BaseChatModel = BaseChatModel
    module.Document = Document
    module.LLMChain = LLMChain
    return module


class CustomRetrieverReflectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_mod = _load_agent_module()

    def _agent(self, retriever):
        return self.agent_mod.Agent(
            name="parent",
            llm=self.agent_mod.BaseChatModel(),
            long_term_memory=retriever,
        )

    def test_missing_memory_stream_skips_reflection_without_raising(self):
        class PlainRetriever(self.agent_mod.BaseRetriever):
            def add_documents(self, documents, **kwargs):
                raise AssertionError("reflection must not write memories")

        agent = self._agent(PlainRetriever())
        agent.status = "running"
        self.assertEqual(agent._synthesize_memories(), [])
        self.assertEqual(agent.pause_to_reflect(), [])
        self.assertEqual(agent.status, "running")
        self.assertEqual(agent.cumulative_strength, 0.0)

    def test_time_weighted_stream_still_synthesizes_recent_memories(self):
        Document = self.agent_mod.Document

        class StreamRetriever(self.agent_mod.BaseRetriever):
            def __init__(self):
                self.memory_stream = [
                    Document(page_content="older"),
                    Document(page_content="newer"),
                ]

        class Synthesis(self.agent_mod.LLMChain):
            def run(self, **kwargs):
                self.seen = kwargs["memories"]
                return ["synthesized"]

        agent = self._agent(StreamRetriever())
        synthesis = Synthesis()
        agent.synthesis = synthesis
        self.assertEqual(agent._synthesize_memories(last_k=1), ["synthesized"])
        self.assertEqual(synthesis.seen, "newer")


if __name__ == "__main__":
    unittest.main()
