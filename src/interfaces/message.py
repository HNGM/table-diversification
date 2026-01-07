from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from openai.types.beta.thread_create_params import MessageAttachment
from openai.types.beta.threads import MessageContentPartParam

class Message(BaseModel):
    content: Union[str, List[MessageContentPartParam]]
    role: str
    attachments: List[MessageAttachment] = []
    
    def to_openapi_format(self):
        return {
            "role": self.role,
            "content": self.content
        }

class UserMessage(Message):
    role: Literal["user"]

    def __init__(
        self,
        content:Union[str, List[MessageContentPartParam]],
        attachments: List[MessageAttachment] = None
    ):
        super().__init__(
            content=content,
            attachments=attachments or [],
            role="user"
        )

class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    tool_calls: Optional[List[Any]] = None
    attachments: Optional[List[MessageAttachment]] = []
    content: Optional[str] = None
    def __init__(
        self,
        content: Optional[Union[str, List[MessageContentPartParam]]],
        attachments: Optional[List[MessageAttachment]] = None,
        tool_calls: Optional[List[Any]] = None
    ):
        # Initialize base class first
        super().__init__(
            content=content,
            attachments=attachments or [],
            role="assistant",
            tool_calls=tool_calls
        )
        
    def to_openapi_format(self):
        out = super().to_openapi_format()
        out["content"] =  out["content"] if out["content"] else " "
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        return out
    
class ToolMessage(BaseModel):
    role: Literal["tool"]
    tool_call_id: str
    content: str
    def __init__(
        self,
        content: str,
        tool_call_id: str,
    ):
        super().__init__(
            content=content,
            tool_call_id=tool_call_id,
            role="tool"
        )
    
    def to_openapi_format(self):
        return self.model_dump(mode= "json")
