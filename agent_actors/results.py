from typing import Any

from langchain.schema import AgentAction, AgentFinish


def format_agent_result(result: Any) -> str:
    """Return agent outputs as prompt-safe text for parent/child joins."""
    if isinstance(result, AgentFinish):
        if "result" in result.return_values:
            return str(result.return_values["result"])
        if "output" in result.return_values:
            return str(result.return_values["output"])
        return str(result.return_values)

    if isinstance(result, AgentAction):
        return result.log or f"{result.tool}: {result.tool_input}"

    return str(result)
