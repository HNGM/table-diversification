from abc import ABC, abstractmethod
from typing import Any, Optional, Type, List
from openai import BadRequestError
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

from ..utils.llm_config import LLMConfig

from .message import Message, UserMessage

class AgentResponse(BaseModel):
    RawResponse: Any
    ParsedResponse: Any = Field(init=False)

    @model_validator(mode = "before")
    @classmethod
    def populate_parsed_response(cls, dic):
        dic["ParsedResponse"] = cls._parse_raw_response(dic["RawResponse"])
        return dic
    
    def _parse_raw_response(raw_response: str) -> str:
        """Default implementation that returns the raw response unchanged"""
        if raw_response == "":
            raise Exception("Received empty response from agent")
        return raw_response

class Agent(ABC):
    def __init__(
        self,
        llm_config: LLMConfig,
        name: str,
        response_type: Type[AgentResponse] = AgentResponse
    ):
        self.llm_config = llm_config
        self.client = llm_config.get_client()
        self.name = name
        self.response_type: Type[AgentResponse] = response_type
        self._max_response_failure_tries = 5

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def get_agent_name(self):
        return self.name

    def run(self, messages: Optional[list[Message]] = None) -> AgentResponse:
        tries = self._max_response_failure_tries
        self._save_state()
        while tries > 0:
            try:
                response = self.response_type(RawResponse=self._run(messages))
                break
            except BadRequestError as e:
                self._restore_last_state()
                raise Exception("Failed to get valid response from agent on endpoint: ", self.llm_config.endpoint, "\nThe following exception was raised: ", str(e))
            except Exception as e:
                self._restore_last_state()
                tries -= 1
        if tries == 0:
            raise Exception("Failed to get valid response from agent on endpoint: ", self.llm_config.endpoint)
        return response

    @abstractmethod
    def _run(self, messages: Optional[list[Message]]) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass

    def reset_client(self):
        old_client = self.client
        self.client = self.llm_config.get_client()
        old_client.close()
    
    @abstractmethod
    def _save_state(self):
        pass
    
    @abstractmethod
    def _restore_last_state(self):
        pass   

    @abstractmethod
    def upload_files(self, files:List[Path], metadata:str) -> 'UserMessage':
        pass
