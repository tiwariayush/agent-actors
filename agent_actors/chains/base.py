import json
from typing import Any, Dict

from langchain import LLMChain


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Plan/Adjust prompts ask for "just the JSON", but models commonly append a
        # short explanation after a valid value. json.loads rejects that trailing
        # text ("Extra data") and aborts the parent run; raw_decode accepts the
        # leading JSON document and ignores the remainder.
        text = super()._call(inputs)["json"].strip()
        data, _ = json.JSONDecoder().raw_decode(text)
        return {"json": data}
