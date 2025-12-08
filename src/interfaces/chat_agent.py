from abc import abstractmethod
import json
from time import sleep
from typing import Any, Optional, Type, List, Dict
from pathlib import Path
from openai import BadRequestError
from pydantic import BaseModel
import base64
from openai.types.beta.threads import TextContentBlockParam, ImageFileContentBlockParam, ImageURLContentBlockParam, MessageContentPartParam

from ..interfaces.message import ToolMessage
from ..utils.llm_config import LLMConfig
from ..interfaces import Agent, AgentResponse, Message, AssistantMessage, UserMessage
# import backoff

class ChatAgentConfig(BaseModel):
    model: str
    name: Optional[str] = None
    instructions : Optional[str] = None
    tool_desc : Optional[Any] = None
    tool_map: Optional[Dict[str, Any]] = None


class ChatAgent(Agent):
    def __init__(
        self,
        llm_config: LLMConfig,
        config: ChatAgentConfig,
        response_type: Type[AgentResponse] = AgentResponse,
        max_tries: int = 8
    ):
        super().__init__(llm_config, config.name, response_type)
        self.messages : list['Message'] = []
        self.config : ChatAgentConfig = config
        self._reserve_state: list['Message'] = []
        self.file_id_path_map : Dict[str, Path] = {}
        self._max_tries = max_tries

    def _get_init_message_content(self):
        pass

    def _run(self, messages: Optional[list['Message']] = None) -> str:
        if len(self.messages) == 0:
            self._get_init_message_content()

        if messages:
            self.messages = self.messages + messages

        for i in range(self._max_tries):
            try:
                for j in range(self._max_tries):
                    message = self.llm_config.chat_request(messages= self.messages, tools=self.config.tool_desc)
                    if message.tool_calls:
                        execution_output = []
                        for tool_call in message.tool_calls:
                            try:
                                tool_output = self.config.tool_map[tool_call.function.name](**json.loads(tool_call.function.arguments))
                                execution_output.append((tool_call, tool_output))
                            except Exception as e:
                                execution_output.append((tool_call, "Tool call failed with exception: " + str(e)))
                                continue
                        tool_calls = [tool_call.model_dump(mode= "json") for tool_call, tool_output in execution_output]
                        if len(tool_calls) > 0:
                            self.messages.append(AssistantMessage(content=message.content, tool_calls=tool_calls))
                            for tool_call, tool_output in execution_output:
                                self.messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call.id))
                    else:
                        response = message.content
                        if response is None:
                            raise Exception("Failed to get valid response from agent on endpoint: ", self.llm_config.endpoint)
                        break
                if j == self._max_tries - 1:
                    raise Exception("Failed to get valid response from agent on endpoint: ", self.llm_config.endpoint)
                break
            except BadRequestError as e:
                raise e
            except Exception as e:
                sleep(2**i)
               
        if i == self._max_tries - 1:
            raise Exception("Failed to get valid response from agent on endpoint: ", self.llm_config.endpoint) 

        self.messages.append(AssistantMessage(response))
        return self.messages[-1].content

    def reset(self):
        self._reset_history()

    def _reset_history(self):
        self.messages = []

    def _save_state(self):
        self._reserve_state = [messages for messages in self.messages]
    
    def _restore_last_state(self):
        self.messages = self._reserve_state
    
    def upload_files(self, files:List[Path], metadata:str="")->'UserMessage':
        'Supports uploading only image files'
        messages : List[MessageContentPartParam] = []

        for file in files:
            if file.suffix not in ['.png', '.jpg', '.jpeg']:
                raise ValueError(f"file type {file.suffix} not supported for upload in chat agent")

            else:
                with open(file, "rb") as f:
                    file_data = f.read()

            try:
                uploaded_file = self.client.files.create(file=file_data, purpose="vision")
                self.file_id_path_map[uploaded_file.id] = file
                image_file_dict = {
                    "type": "image_file",
                    "image_file": {
                        "file_id": uploaded_file.id
                    }
                }
                message = f"The above Image file: - File ID: {uploaded_file.id} | File Name: {file.name}\n"
                message_dict = {
                    "type": "text",
                    "text": message
                }
                messages.append(ImageFileContentBlockParam(image_file_dict))
                messages.append(TextContentBlockParam(message_dict))

            except:
                encoded_file = base64.b64encode(file_data).decode("ascii")
                image_url_dict = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{file.suffix[1:]};base64,{encoded_file}"
                    }
                }
                messages.append(ImageURLContentBlockParam(image_url_dict))

        metadata_dict = {
            "type": "text",
            "text": metadata
        }
        messages.append(TextContentBlockParam(metadata_dict))

        return UserMessage(content=messages)
    
    def delete_file(self, file_id):
        self.client.files.delete(file_id= file_id)
        del self.file_id_path_map[file_id]
    
    def __del__(self):
        # delete the files if they have been uploaded
        existing_file_ids = [file_id for file_id in self.file_id_path_map.keys()]
        for file_id in existing_file_ids:
            self.delete_file(file_id)
