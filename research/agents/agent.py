import json
from pathlib import Path
from research.agents.utils.code_tool import CodeTool
from src.interfaces.ada_agent import AdaAgent, AdaAgentConfig
from src.interfaces.chat_agent import ChatAgent, ChatAgentConfig
from src.interfaces.message import UserMessage, Message
from src.utils.llm_config import LLMConfig, load_llm_configs
from typing import Optional, Type, Union
from research.agents.utils.model_response import JsonResponseParser
from src.interfaces import AgentResponse

class FunctionCallAdaAgentResponse(AgentResponse):
    
    @classmethod
    def _parse_raw_response(cls, raw_response):
        raw_response = json.loads(raw_response)
        return JsonResponseParser._parse_raw_response(raw_response["response"])

class FunctionCallADAConfig(AdaAgentConfig):
    
    def __init__(self, model: str, llm_config_path: Path):
        super().__init__(framework=FunctionCallAdaAgent, model=model, llm_config_path=llm_config_path)
    
    def load_model(self, index:int):
        llm_configs = load_llm_configs(self.llm_config_path, self.model)
        llm_config = llm_configs[index%len(llm_configs)]
        ada_agent = self.framework(llm_config)
        return ada_agent

class FunctionCallAdaAgent(ChatAgent, AdaAgent):
    def __init__(
        self, 
        llm_config: LLMConfig,
        prompt: str
    ):
        self.sandbox = CodeTool()
        AdaAgent.__init__(self, response_type=FunctionCallAdaAgentResponse)
        ChatAgent.__init__(self,
            llm_config=llm_config,
            config=ChatAgentConfig(
                model=llm_config.model,
                name="FunctionCallAdaAgent",
                instructions=prompt,
                tool_desc = [{
                "type": "function",
                "function":{
                    "name": "CodeTool",
                    "description": self.sandbox.description,
                    "parameters":{
                        "type": "object",
                        "properties":{
                            "code_str": {"type": "string"}
                        },
                        "required": ["code_str"] 
                    }        
                }
            }],
                tool_map= {"CodeTool": lambda code_str: self.sandbox.run_code(code_str).log}
            ),
            response_type=FunctionCallAdaAgentResponse
        )
    
    def _get_init_message_content(self):
        self.messages.append(Message(role="system", content=self.config.instructions))
        
    def upload_files(self, files, metadata):
        self.sandbox.upload_files(files)
        return UserMessage(content=metadata)
    
    def upload_image_files(self, image_files, metadata):
        return ChatAgent.upload_files(self, image_files, metadata)
    
    def download_generated_images(self, save_dir):
        return self.sandbox.download_generated_images(save_dir)
    
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
    

