import json
from pathlib import Path
from research.agents.utils.code_tool import CodeTool
from src.interfaces.ada_agent import AdaAgent
from src.interfaces.chat_agent import ChatAgent, ChatAgentConfig
from src.interfaces.message import UserMessage, Message
from src.utils.llm_config import LLMConfig, load_llm_configs
from typing import Optional, Type, Union
from research.agents.utils.model_response import JsonResponseParser
from src.interfaces import AgentResponse

class NoFunctionCallAdaAgentResponse(AgentResponse):
    
    @classmethod
    def _parse_raw_response(cls, raw_response):
        raw_response = json.loads(raw_response)
        return JsonResponseParser._parse_raw_response(raw_response["response"])

class NoFunctionCallAdaAgent(ChatAgent, AdaAgent):
    def __init__(
        self, 
        llm_config: LLMConfig,
        prompt: str
    ):
        AdaAgent.__init__(self, response_type=NoFunctionCallAdaAgentResponse)
        ChatAgent.__init__(self,
            llm_config=llm_config,
            config=ChatAgentConfig(
                model=llm_config.model,
                name="NoFunctionCallAdaAgent",
                instructions=prompt,
            ),
            response_type=NoFunctionCallAdaAgentResponse
        )
    
    def _get_init_message_content(self):
        self.messages.append(Message(role="system", content=self.config.instructions))
        
    def upload_files(self, files, metadata):
        raise NotImplementedError("NoFunctionCallAdaAgent does not support file upload.")
    
    def upload_image_files(self, image_files, metadata):
        return ChatAgent.upload_files(self, image_files, metadata)
    
    def download_generated_images(self, save_dir):
        raise NotImplementedError("NoFunctionCallAdaAgent does not support image download.")

    def _get_monologues(self):
        monologue = []
        for msg in reversed(self.messages):
            if msg.role == "user":
                break
            monologue.append(msg.model_dump(mode="json"))
        return [m for m in reversed(monologue)]
    
    def _run(self, messages: Optional[list[Union[str, Message]]]) -> str:
        response = ChatAgent._run(self, messages)
        return json.dumps({
            "monologue": self._get_monologues(),
            "response": response
        })
    

