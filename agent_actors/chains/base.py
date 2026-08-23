import json
from typing import Any, Dict

from langchain import LLMChain


def _replace_unquoted_angle_placeholders(text: str) -> str:
    """Replace unquoted <...> tokens copied from Plan/Adjust examples.

    Both prompts show dummy values such as ``<int>`` and
    ``<synthesized result satisfying objective>`` without quotes. Those
    tokens are invalid JSON, so json.loads raises JSONDecodeError.
    Placeholders already inside strings are left unchanged.
    """
    out = []
    in_string = False
    escape = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "<":
            close = text.find(">", i + 1)
            if close != -1 and "\n" not in text[i:close]:
                out.append("null")
                i = close + 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _loads_plan_adjust_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _replace_unquoted_angle_placeholders(text)
        if repaired == text:
            raise
        return json.loads(repaired)


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = super()._call(inputs)["json"].strip()
        data = _loads_plan_adjust_json(text)
        return {"json": data}
