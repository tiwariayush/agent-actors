import json
from typing import Any, Dict

from langchain import LLMChain


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape literal control characters that appear inside JSON strings.

    Plan/Adjust models often put real newlines (and other U+0000–U+001F
    controls) inside ``result``, ``speak``, or ``task`` strings. Those
    characters are illegal in JSON strings, so json.loads raises
    JSONDecodeError. Whitespace outside strings is left unchanged.
    Already-escaped sequences such as ``\\n`` are also left unchanged.
    """
    out = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            code = ord(ch)
            if code < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{code:04x}")
                continue
            out.append(ch)
            continue

        if ch == '"':
            in_string = True
        out.append(ch)

    return "".join(out)


def _loads_plan_adjust_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _escape_control_chars_in_strings(text)
        if repaired == text:
            raise
        return json.loads(repaired)


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = super()._call(inputs)["json"].strip()
        data = _loads_plan_adjust_json(text)
        return {"json": data}
