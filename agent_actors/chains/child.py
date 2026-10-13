from textwrap import dedent
from typing import List

from langchain import LLMChain, MRKLChain, PromptTemplate
from langchain.agents import Tool, ZeroShotAgent
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AgentFinish, OutputParserException


class ReActAgent(ZeroShotAgent):
    @classmethod
    def from_llm_and_tools(
        cls,
        **kwargs,
    ) -> LLMChain:
        return super().from_llm_and_tools(
            **kwargs,
            prefix="Complete the task below. Be very specific. You have access to the following tools:",
            format_instructions=dedent(
                """\
                Use the following format.

                Task: the input task you must complete
                Thought: you should always think about what to do
                Reasoning: the reasoning behind your thought
                Action: the action to take, should be one of {tool_names}, and only the tool name
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Reasoning/Action/Action Input/Observation cycle can repeat N times)
                Thought: The task has been completed appropriately and accurately
                Final Answer: the final result of this task, directly, not summarized in any way

                When you're done your work, or you have the answer, start by saying "Final Answer: "
                """
            ),
            suffix=dedent(
                """\
                {context}

                Task: {task}
                Thought: {agent_scratchpad}\
                """
            ),
            input_variables=[
                "context",
                "task",
                "agent_scratchpad",
            ],
        )

    def return_stopped_response(
        self, early_stopping_method, intermediate_steps, **kwargs
    ):
        """Finish after the iteration limit without dropping completed work.

        ``Agent.return_stopped_response`` asks the model for a final answer, then
        falls back to the raw reply only when ``parse`` *returns* a tool action.
        ``MRKLOutputParser`` instead raises ``OutputParserException`` for a plain
        sentence. That exception escapes ``ChildAgent.run``, fails the Ray task,
        and makes ``ParentAgent`` abort in ``ray.get`` before Adjust — discarding
        sibling results that already finished.
        """
        if early_stopping_method != "generate":
            return super().return_stopped_response(
                early_stopping_method, intermediate_steps, **kwargs
            )

        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        thoughts += "\n\nI now need to return a final answer based on the previous steps:"
        full_inputs = {
            **kwargs,
            "agent_scratchpad": thoughts,
            "stop": self._stop,
        }
        full_output = self.llm_chain.predict(**full_inputs)
        try:
            parsed_output = self.output_parser.parse(full_output)
        except OutputParserException:
            return AgentFinish({"output": full_output}, full_output)
        if isinstance(parsed_output, AgentFinish):
            return parsed_output
        return AgentFinish({"output": full_output}, full_output)


class Do(MRKLChain):
    max_iterations = 3
    early_stopping_method = "generate"
    return_intermediate_steps = True

    @classmethod
    def from_llm(
        cls, llm: BaseChatModel, tools: List[Tool], verbose: bool = True, **kwargs
    ) -> LLMChain:
        agent = ReActAgent.from_llm_and_tools(
            **kwargs,
            llm=llm,
            tools=tools,
            verbose=verbose,
        )
        return cls.from_agent_and_tools(
            agent, tools, callback_manager=kwargs.get("callback_manager", None)
        )


class Check(LLMChain):
    @classmethod
    def from_llm(cls, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    You are evaluating the result of an AI agent against its task to verify its accurate compmletion.

                    Here is the relevant context: {context}

                    Here is the result: {learning}

                    If the result accomplishes its task, return "Complete".
                    Otherwise, return new task for the agent to complete.

                    Do not embellish.

                    Task: \
                    """
                ),
            ),
        )
