import json
import random
from typing import Literal, Optional
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, model_validator
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import traceback
try:
    from prose.substrate import Client
except:
    print("Error importing prose.substrate. Run `pip install -e C:\prose\etc\PythonPackages\substrate` to install")
    print("Importing default substrate client")
    from .substrate import LLMClient

_OAI_API_VERSION = "2024-09-01-preview"

class LLMConfig(BaseModel):
    endpoint_mode: Literal["azure_oai", "openai", "azure_ml", "gemini", "substrate"]
    model: str
    deployment_name: Optional[str]
    api_version: str = _OAI_API_VERSION 
    ado_auth: Optional[bool] = True
    endpoint: Optional[str] = None
    api_key: Optional[str] = None

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
                return AzureOpenAI(**aoai_config)
            elif self.endpoint_mode == "substrate":
                try:
                    return Client(None)
                except:
                    return LLMClient(self.endpoint)
            base_url = self.endpoint if self.endpoint else None
            return OpenAI(api_key=self.api_key, base_url=base_url)
            
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
            try:
                response = client.one(req_args["model"], req_args)
                response = ChatCompletion(**response)
            except:
                response = client.send_request(req_args["model"], req_args)
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