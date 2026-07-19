import json
import re
from typing import Any, Dict

from langchain import LLMChain


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = super()._call(inputs)["json"].strip()
        fenced_json = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
        )
        if fenced_json:
            text = fenced_json.group(1)
        data = json.loads(text)
        return {"json": data}
