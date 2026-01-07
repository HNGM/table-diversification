from src.interfaces import ChatAgent, ChatAgentConfig, AgentResponse, Message
from src.utils.llm_config import LLMConfig
from research.agents.utils.model_response import JsonResponseParser
from typing import Type

class InstrChatAgent(ChatAgent):
    def __init__(
        self,
        llm_config: LLMConfig,
        prompt: str = "",
        response_type: Type[AgentResponse] = AgentResponse
    ):
        super().__init__(
            llm_config,
            ChatAgentConfig(
                model=llm_config.deployment_name, 
                name="InstrChatAgent",
                instructions=prompt
            ), 
            response_type=response_type
        )
    
    def _get_init_message_content(self):
        self.messages.append(Message(role="system", content=self.config.instructions))

    def _reset_history(self):
        self.messages = [Message(role="system", content=self.config.instructions)]