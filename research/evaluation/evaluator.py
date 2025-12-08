import re
from typing import List, Any, Union
from pathlib import Path
import json

from src.interfaces import AgentResponse, ChatAgent, ChatAgentConfig, Message, UserMessage
from src.utils.llm_config import LLMConfig

SYSTEM = """
You are tasked with evaluating a data analysis assistant system Dazza. 
Your main task will be to evaluate if Dazza's answer to the question is correct or not.

Given below are inputs required for you to make a call on whether Dazza's response is correct or not.

QUESTION: a complex data analytics question which the user asks Dazza to solve.

ANSWER: A reference answer or guideline to solve the query which complements the actual solution.

**Dazza's Response**
DAZZA: Response from Dazza which you should verify if it is answering the user's question or not. 

Here are some additional rules to follow:
- If Dazza fails to provide an answer or a response, return False.
- Your comparison should be qualitative and not quantitative. You should look for exact match only if it matters in the context of the QUESTION.
For example, If the question asks for top 5 values, you should check if Dazza's response also provides top 5 values that match with the actual solution. However if the question asks for top values and Dazza provides top 10 instead of top 5, do not penalise it as long as the 5 values exist in the top 10. Apply a similar logic when comparing graphs.
- You can also consider qualitative match to be a success if Dazza's response provides a subset of the actual solution. For example, if the actual solution contains 5 values and Dazza's response contains 3 of those 5 values, you can consider it to be a qualitative match. This logic can be applied to plots as well.

--OUTPUT FORMAT--
Your output should strictly be a json dictionary. You will be heavily penalized if you fail to follow this format.
```json
{
    "actual_solution": str <following step 1>,
    "dazza_response": str <following step 2>,
    "verdict": true/false,
    "reason": str <explain why you think there was or was not a qualitative match.>
}
```
"""

class QualitativeEvaluatorResponse(AgentResponse):
    @classmethod
    def _parse_raw_response(cls, raw_response):
        pattern = r"```json\n(.*?)```"
        matches = re.findall(pattern, raw_response, re.DOTALL)
        if len(matches) == 0:
            raise Exception("Incorrect response format from Qualitative Evaluator")
        response_dict = json.loads(matches[0])
        if isinstance(response_dict['verdict'], str):
            if "true" in response_dict['verdict'].lower():
                response_dict['verdict'] = True
            else:
                response_dict['verdict'] = False
        return response_dict
    
class QualitativeEvaluator(ChatAgent):
    def __init__(
        self,
        llm_config: LLMConfig,
    ):
        super().__init__(
            llm_config,
            ChatAgentConfig(
                model=llm_config.deployment_name,
                name="QualitativeEvaluator",
                instructions=SYSTEM
            ),
            response_type=QualitativeEvaluatorResponse
        )
    
    def _get_init_message_content(self):
        self.messages.append(Message(role="system", content=self.config.instructions))

    def evaluate(
            self, 
            query:str, 
            answer:str,
            tool_response:str,
        )->dict:
        input_message = f"Given below are inputs required for you to make a call on whether Dazza's response qualitatively match with the solution or not.\n"
        input_message += f"QUESTION: {query}\n"
        input_message += f"ANSWER: {answer}\n"
        input_message += f"DAZZA: {tool_response}\n"
        messages = [UserMessage(content=input_message)]
        output = self.run(messages=messages)

        return output.ParsedResponse
    
    

