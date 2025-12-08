from src.interfaces import AgentResponse
import re
import json
import ast

class JsonResponseParser(AgentResponse):
    def _parse_raw_response(raw_response:str)->str:
        pattern = r"```json\n(.*?)```"
        matches = re.findall(pattern, raw_response, re.DOTALL)
        if len(matches) == 0:
            raise Exception("Incorrect response format from the agent") 
        try:
            return json.loads(matches[0])  
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(matches[0])
            except ValueError:
                raise Exception("Failed to parse the response as JSON or Python literal")