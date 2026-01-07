from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Type, Optional
from pydantic import BaseModel

from . import AgentResponse, Message, UserMessage
from ..utils.llm_config import load_llm_configs

class AdaAgent(ABC):
    def __init__(
        self,
        response_type: Type[AgentResponse] = AgentResponse
    ):
        self.response_type = response_type

    @abstractmethod
    def upload_files(self, files: List[Path], metadata:str) -> 'UserMessage':
        pass

    @abstractmethod
    def download_generated_images(self, save_dir: Union[str, Path]) -> List[Path]:
        pass

    def run(self, messages: Optional[list[Union[str, Message]]]) -> AgentResponse:
        return self.response_type(RawResponse=self._run(messages))
    
    @abstractmethod
    def _run(self, messages: Optional[list[Union[str, Message]]]) -> str:
        pass

