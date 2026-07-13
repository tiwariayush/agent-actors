from typing import Any

from langchain.schema import AgentAction, AgentFinish


def format_agent_result(result: Any) -> str:
    if isinstance(result, AgentFinish):
        values = result.return_values
        if isinstance(values, dict):
            for key in ("result", "output", "speak"):
                if key in values:
                    return str(values[key])
        return str(values)

    if isinstance(result, AgentAction):
        return result.log

    return str(result)
