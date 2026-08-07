import json
from typing import Any, Dict

from langchain import LLMChain


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Plan/Adjust expect a JSON array/object, but models sometimes emit a
        # JSON string whose contents are that document (double-encoded). One
        # json.loads then yields a str; ParentAgent.run iterates Plan chars or
        # indexes Adjust like a mapping and crashes. Decode once more when the
        # first parse is a string.
        data = json.loads(super()._call(inputs)["json"].strip())
        if isinstance(data, str):
            data = json.loads(data)
        return {"json": data}
