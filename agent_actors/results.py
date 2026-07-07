def format_agent_result(result) -> str:
    """Convert LangChain agent results into prompt-safe text."""
    if isinstance(result, str):
        return result

    return_values = getattr(result, "return_values", None)
    if isinstance(return_values, dict):
        for key in ("output", "result", "speak"):
            if return_values.get(key) is not None:
                return str(return_values[key])
        return str(return_values)

    log = getattr(result, "log", None)
    if log is not None:
        return str(log)

    return str(result)
