from src.interfaces import AgentResponse
import re
import json
import ast

class JsonResponseParser(AgentResponse):
    def _parse_raw_response(raw_response:str)->str:
        if "```json" not in raw_response:
            match = raw_response
        else:
            pattern = r"```json\n(.*?)```"
            matches = re.findall(pattern, raw_response, re.DOTALL)
            if len(matches) == 0:
                raise Exception(f"Incorrect response format from the agent: {raw_response}") 
            match = matches[0]
        try:
            return json.loads(match)  
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(match)
            except ValueError:
                raise Exception(f"Failed to parse the response as JSON or Python literal: {raw_response}")