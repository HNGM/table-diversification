import json
import random
from typing import Literal, Optional
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel, model_validator
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import traceback
from .substrate import LLMClient
from src.utils.huggingface import HuggingFaceClient


_OAI_API_VERSION = "2024-09-01-preview"

class LLMConfig(BaseModel):
    endpoint_mode: Literal["azure_oai", "openai", "azure_ml", "gemini", "substrate", "huggingface"]
    model: str
    deployment_name: Optional[str]
    api_version: str = _OAI_API_VERSION
    ado_auth: Optional[bool] = True
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    _client_cache: Optional[object] = None

    @model_validator(mode='before')
    @classmethod
    def validate_data(cls, data: dict):
        if data['endpoint_mode'] == 'openai' and data.get('api_key') is None:
            raise ValueError("Azure endpoint requires an api key")
        if data['endpoint_mode'] == 'azure_oai' and not data.get('ado_auth') and data.get("api_key") is None:
            raise ValueError("ADO auth or API key is required for Azure endpoint")            
        if data.get('deployment_name') is None:
            data['deployment_name'] = data['model']
        return data
    
    def get_client(self):
        if self._client_cache is not None:
            return self._client_cache
        try:
            if self.endpoint_mode == "azure_oai":
                aoai_config = {
                    "api_version": self.api_version,
                    "azure_endpoint": self.endpoint
                }
                if self.ado_auth:
                    token = get_bearer_token_provider(
                        DefaultAzureCredential(),
                        "https://cognitiveservices.azure.com/.default"
                    )
                    aoai_config["azure_ad_token_provider"] = token
                else:
                    aoai_config["api_key"] = self.api_key
                client = AzureOpenAI(**aoai_config)
            elif self.endpoint_mode == "substrate":
                client = LLMClient(self.endpoint)
            elif self.endpoint_mode == "huggingface":
                client = HuggingFaceClient(self.deployment_name if self.deployment_name else self.model)
            else:
                base_url = self.endpoint if self.endpoint else None
                client = OpenAI(api_key=self.api_key, base_url=base_url)
            
            self._client_cache = client
            return client
            
        except Exception as e:
            print(self.model_dump_json(indent = 4))
            print(traceback.format_exc())
            print("Error in getting client", e)
    
    def chat_request(self, **kwargs):
        client = self.get_client()
        
        req_args = {}
        req_args["model"] = self.deployment_name if self.deployment_name else self.model
        if self.endpoint_mode == "azure_oai" or self.endpoint_mode == "openai":
            req_args["messages"] = kwargs["messages"]
        else:
            req_args["messages"] = [msg.to_openapi_format() for msg in kwargs["messages"]]
        if "tools" in kwargs and kwargs["tools"]:
            req_args["tools"] = kwargs["tools"]
        if self.endpoint_mode == "substrate":
            response = client.send_request(req_args["model"], req_args)
        elif self.endpoint_mode == "huggingface":
            response = client.send_request(req_args)
            return ChatCompletionMessage(content=response, role="assistant")
        else:        
            response = client.chat.completions.create(
                    **req_args
                )
        return response.choices[0].message
 
def load_llm_configs(path, use_model: Optional[str] = None):    
    check_model = use_model is not None
    with open(path) as file:
        configs = json.load(file)
    if isinstance(configs, dict):
        llm_configs =  [LLMConfig(**configs)]
    else:
        llm_configs = [LLMConfig(**config) for config in configs ]
    random.shuffle(llm_configs)
    return [config for config in llm_configs if check_model and config.model == use_model]