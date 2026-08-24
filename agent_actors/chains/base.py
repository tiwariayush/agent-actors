import json
from typing import Any, Dict

from langchain import LLMChain

_CONFIDENCE_TOKEN = "confidence"


def _replace_unquoted_confidence_identifier(text: str) -> str:
    """Replace the Adjust example's unquoted ``confidence`` value.

    The Adjust prompt shows ``"confidence": confidence`` (a bare
    identifier). Models copy that token, and json.loads raises
    JSONDecodeError after child work has already finished. The
    replacement is ``0`` so ParentAgent.run's ``adjustment["confidence"] >= 8``
    comparison on main asks for human input instead of TypeError.
    Occurrences inside JSON strings (including the ``"confidence"`` key)
    are left unchanged.
    """
    out = []
    in_string = False
    escape = False
    i = 0
    n = len(text)
    token_len = len(_CONFIDENCE_TOKEN)

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

        if text.startswith(_CONFIDENCE_TOKEN, i):
            end = i + token_len
            prev = text[i - 1] if i else ""
            nxt = text[end] if end < n else ""
            prev_ok = not (prev.isalnum() or prev == "_")
            next_ok = not (nxt.isalnum() or nxt == "_")
            if prev_ok and next_ok:
                out.append("0")
                i = end
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _loads_plan_adjust_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _replace_unquoted_confidence_identifier(text)
        if repaired == text:
            raise
        return json.loads(repaired)


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = super()._call(inputs)["json"].strip()
        data = _loads_plan_adjust_json(text)
        return {"json": data}
