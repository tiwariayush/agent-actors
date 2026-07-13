import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def install_dependency_stubs():
    for module_name in list(sys.modules):
        if module_name == "agent_actors" or module_name.startswith("agent_actors."):
            sys.modules.pop(module_name)

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    ray = types.ModuleType("ray")

    class ActorHandle:
        def __init__(self, instance):
            self._instance = instance

        def __getattr__(self, name):
            attribute = getattr(self._instance, name)
            if callable(attribute):
                return types.SimpleNamespace(
                    remote=lambda *args, **kwargs: attribute(*args, **kwargs)
                )
            return attribute

    def remote(*decorator_args, **decorator_kwargs):
        def decorate(cls):
            cls.remote = classmethod(
                lambda actor_cls, *args, **kwargs: ActorHandle(
                    actor_cls(*args, **kwargs)
                )
            )
            return cls

        if decorator_args and isinstance(decorator_args[0], type):
            return decorate(decorator_args[0])
        return decorate

    ray.remote = remote
    ray.get = lambda refs: refs
    ray.wait = lambda refs, num_returns=1: (refs[:num_returns], refs[num_returns:])
    ray.ObjectRef = object
    sys.modules["ray"] = ray

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, *args, **kwargs):
            if args:
                raise TypeError("positional args are not supported by this stub")
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    langchain = types.ModuleType("langchain")

    class LLMChain:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def run(self, **kwargs):
            return getattr(self, "response", "")

        def __call__(self, *args, **kwargs):
            return {}

    class MRKLChain(LLMChain):
        @classmethod
        def from_agent_and_tools(cls, *args, **kwargs):
            return cls(*args, **kwargs)

    class PromptTemplate:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def from_template(cls, *args, **kwargs):
            return cls(*args, **kwargs)

    langchain.LLMChain = LLMChain
    langchain.MRKLChain = MRKLChain
    langchain.PromptTemplate = PromptTemplate
    sys.modules["langchain"] = langchain

    agents = types.ModuleType("langchain.agents")

    class Tool:
        pass

    class ZeroShotAgent:
        @classmethod
        def from_llm_and_tools(cls, *args, **kwargs):
            return cls()

    agents.Tool = Tool
    agents.ZeroShotAgent = ZeroShotAgent
    sys.modules["langchain.agents"] = agents

    chat_models = types.ModuleType("langchain.chat_models")
    chat_models_base = types.ModuleType("langchain.chat_models.base")

    class BaseChatModel:
        pass

    chat_models_base.BaseChatModel = BaseChatModel
    sys.modules["langchain.chat_models"] = chat_models
    sys.modules["langchain.chat_models.base"] = chat_models_base

    schema = types.ModuleType("langchain.schema")

    class AgentAction:
        def __init__(self, tool, tool_input, log):
            self.tool = tool
            self.tool_input = tool_input
            self.log = log

    class AgentFinish:
        def __init__(self, return_values, log):
            self.return_values = return_values
            self.log = log

    class BaseRetriever:
        pass

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema.AgentAction = AgentAction
    schema.AgentFinish = AgentFinish
    schema.BaseRetriever = BaseRetriever
    schema.Document = Document
    sys.modules["langchain.schema"] = schema


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime_modules():
    install_dependency_stubs()
    modules = {}
    for name, path in [
        ("agent_actors.actors", "agent_actors/actors.py"),
        ("agent_actors.chains.agent", "agent_actors/chains/agent.py"),
        ("agent_actors.chains.base", "agent_actors/chains/base.py"),
        ("agent_actors.chains.child", "agent_actors/chains/child.py"),
        ("agent_actors.chains.parent", "agent_actors/chains/parent.py"),
        ("agent_actors.models", "agent_actors/models.py"),
        ("agent_actors.results", "agent_actors/results.py"),
        ("agent_actors.agent", "agent_actors/agent.py"),
        ("agent_actors.child", "agent_actors/child.py"),
        ("agent_actors.parent", "agent_actors/parent.py"),
    ]:
        modules[name] = load_module(name, path)
    return modules


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        modules = load_runtime_modules()
        actors = modules["agent_actors.actors"]

        class Chain:
            def run(self, *args, **kwargs):
                return "chain", args, kwargs

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", flag=True),
            ("chain", ("task",), {"flag": True}),
        )

    def test_memory_strength_parses_labelled_and_two_digit_scores(self):
        modules = load_runtime_modules()
        agent_module = modules["agent_actors.agent"]
        agent = object.__new__(agent_module.Agent)

        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **kwargs):
                return self.response

        agent.memory_strength = MemoryStrength("Relevance: 8")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

        agent.memory_strength = MemoryStrength("10")
        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

        agent.memory_strength = MemoryStrength("Relevance: 12")
        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

        agent.memory_strength = MemoryStrength("no score")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.0)

    def test_result_formatter_handles_structured_agent_outputs(self):
        modules = load_runtime_modules()
        results = modules["agent_actors.results"]
        schema = sys.modules["langchain.schema"]

        self.assertEqual(
            results.format_agent_result(
                schema.AgentFinish({"confidence": 9, "result": "nested result"}, "")
            ),
            "nested result",
        )
        self.assertEqual(
            results.format_agent_result(schema.AgentAction("Human", "prompt", "ask")),
            "ask",
        )
        self.assertEqual(results.format_agent_result("plain result"), "plain result")

    def test_child_agent_formats_working_memory_before_joining(self):
        modules = load_runtime_modules()
        child_module = modules["agent_actors.child"]
        schema = sys.modules["langchain.schema"]
        child = object.__new__(child_module.ChildAgent)
        child.status = "idle"
        child.max_iterations = 1
        child.get_context = lambda: "child context"
        child.add_memory = lambda memory: None

        class Do:
            def __init__(self):
                self.context = None

            def __call__(self, inputs, return_only_outputs=False):
                self.context = inputs["context"]
                return {"intermediate_steps": [], "output": "child output"}

        class Check:
            def run(self, **kwargs):
                return "Complete"

        child.do = Do()
        child.check = Check()

        result = child.run(
            "child task",
            working_memory=[schema.AgentFinish({"result": "parent result"}, "")],
        )

        self.assertIn("parent result", child.do.context)
        self.assertIn("child output", result)

    def test_parent_agent_formats_nested_results_before_adjusting(self):
        modules = load_runtime_modules()
        parent_module = modules["agent_actors.parent"]
        schema = sys.modules["langchain.schema"]
        parent = object.__new__(parent_module.ParentAgent)
        parent.status = "idle"
        parent.verbose = False
        parent.reflect_every = 10
        parent.children = {}
        parent.get_context = lambda force_refresh=False: "parent context"
        parent.pause_to_reflect = lambda: None

        class Plan:
            def __init__(self):
                self.inputs = None

            def __call__(self, inputs):
                self.inputs = inputs
                return {
                    "json": [
                        {
                            "task_id": 0,
                            "child_id": 0,
                            "task": "nested task",
                            "dependencies": [],
                        }
                    ]
                }

        class Adjust:
            def __init__(self):
                self.results = None

            def run(self, **kwargs):
                self.results = kwargs["results"]
                return {"confidence": 8, "result": "final result"}

        class RemoteRun:
            def remote(self, **kwargs):
                return schema.AgentFinish({"result": "nested result"}, "")

        class Child:
            actor = types.SimpleNamespace(run=RemoteRun())

            def get_context(self):
                return "child context"

        parent.children = {0: Child()}
        parent.plan = Plan()
        parent.adjust = Adjust()

        result = parent.run(
            "parent task",
            working_memory=[schema.AgentFinish({"result": "prior result"}, "")],
        )

        self.assertIn("prior result", parent.plan.inputs["context"])
        self.assertEqual(parent.adjust.results, "nested result")
        self.assertEqual(result.return_values["result"], "final result")

    def test_parent_agent_adds_missing_child_under_planned_id(self):
        modules = load_runtime_modules()
        parent_module = modules["agent_actors.parent"]
        parent = object.__new__(parent_module.ParentAgent)
        parent.status = "idle"
        parent.verbose = False
        parent.reflect_every = 10
        parent.children = {}
        parent.tools = ["tool"]
        parent.llm = object()
        parent.long_term_memory = object()
        parent.get_context = lambda force_refresh=False: "parent context"
        parent.pause_to_reflect = lambda: None

        class Plan:
            callback_manager = "callbacks"

            def __call__(self, inputs):
                return {
                    "json": [
                        {
                            "task_id": 0,
                            "child_id": 42,
                            "task": "dynamic task",
                            "dependencies": [],
                        }
                    ]
                }

        class Adjust:
            def run(self, **kwargs):
                return {"confidence": 8, "result": kwargs["results"]}

        class RemoteRun:
            def remote(self, **kwargs):
                return "dynamic result"

        class FakeChildAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.actor = types.SimpleNamespace(run=RemoteRun())

        parent.plan = Plan()
        parent.adjust = Adjust()
        parent_module.ChildAgent = FakeChildAgent

        result = parent.run("parent task")

        self.assertIn(42, parent.children)
        self.assertEqual(parent.children[42].kwargs["tools"], ["tool"])
        self.assertIs(parent.children[42].kwargs["long_term_memory"], parent.long_term_memory)
        self.assertEqual(result.return_values["result"], "dynamic result")


if __name__ == "__main__":
    unittest.main()
