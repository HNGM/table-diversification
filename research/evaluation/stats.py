import json
from typing import List

def get_tool_call_count(raw_response: str) -> int:
    try:
        monologue = json.loads(raw_response).get("monologue", [])
    except json.decoder.JSONDecodeError:
        monologue = []
    return len([msg for msg in monologue if msg.get("role") == "tool"])

def get_tool_call_count_completion_prosellm(raw_response: List[str]) -> int:
    cnt = 0
    for resp in raw_response:
        if "```python" in resp:
            cnt += 1
    return cnt-1



    





