import json
from typing import Any, Dict

from langchain import LLMChain

# JSON string escapes: ", \, /, b, f, n, r, t, and uXXXX (RFC 8259).
_JSON_SINGLE_ESCAPES = set('"\\/bfnrt')


def _escape_invalid_backslash_in_strings(text: str) -> str:
    """Escape illegal backslash sequences that appear inside JSON strings.

    Plan/Adjust models often copy LaTeX (``\\url``, ``\\cite``), regex
    (``\\d``, ``\\s``), or Windows paths (``\\Users``) into ``result`` /
    ``speak`` / ``task``. Those are not legal JSON escapes, so json.loads
    raises JSONDecodeError. Already-valid escapes (``\\n``, ``\\\\``,
    ``\\uXXXX``) are left unchanged. Backslashes outside strings are left
    unchanged.
    """
    out = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                if i + 1 >= n:
                    out.append("\\\\")
                    i += 1
                    continue
                nxt = text[i + 1]
                if nxt in _JSON_SINGLE_ESCAPES:
                    out.append("\\")
                    out.append(nxt)
                    i += 2
                    continue
                if nxt == "u":
                    hexpart = text[i + 2 : i + 6]
                    if len(hexpart) == 4 and all(
                        c in "0123456789abcdefABCDEF" for c in hexpart
                    ):
                        out.append(text[i : i + 6])
                        i += 6
                        continue
                out.append("\\\\")
                i += 1
                continue
            if ch == '"':
                in_string = False
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_string = True
        out.append(ch)
        i += 1

    return "".join(out)


def _loads_plan_adjust_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _escape_invalid_backslash_in_strings(text)
        if repaired == text:
            raise
        return json.loads(repaired)


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = super()._call(inputs)["json"].strip()
        data = _loads_plan_adjust_json(text)
        return {"json": data}
