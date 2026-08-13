import json
from typing import Any, Dict

from langchain import LLMChain


def _repair_prompt_copied_json(text: str) -> str:
    """Remove artifacts that Plan/Adjust examples encourage models to copy.

    The Plan prompt's JSON example includes a trailing comma and a bare `...`
    placeholder. Those tokens are invalid JSON, so json.loads raises
    JSONDecodeError and ParentAgent.run aborts before dispatch.
    """
    out = []
    in_string = False
    escape = False
    i = 0
    n = len(text)

    def next_nonspace(index: int) -> int:
        while index < n and text[index] in " \t\r\n":
            index += 1
        return index

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

        ellipsis_len = 3 if text.startswith("...", i) else 1 if text.startswith("…", i) else 0
        if ellipsis_len:
            i += ellipsis_len
            j = next_nonspace(i)
            # `[..., task]` — drop the comma that made ellipsis look like an element.
            if j < n and text[j] == ",":
                i = j + 1
            continue

        if ch == ",":
            j = next_nonspace(i + 1)
            skip = 3 if text.startswith("...", j) else 1 if text.startswith("…", j) else 0
            if skip:
                k = next_nonspace(j + skip)
                # Drop "," + ellipsis before a closer; keep the comma if more
                # elements follow (e.g. `[obj, ..., other]`).
                if k < n and text[k] not in "]}":
                    out.append(",")
                i = j + skip
                continue
            if j < n and text[j] in "]}":
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _loads_plan_adjust_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_prompt_copied_json(text)
        if repaired == text:
            raise
        return json.loads(repaired)


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = super()._call(inputs)["json"].strip()
        data = _loads_plan_adjust_json(text)
        return {"json": data}
