import requests
from msal import PublicClientApplication
from openai.types.chat import ChatCompletion
import json
from msal import PublicClientApplication, SerializableTokenCache
import threading
import atexit
import os


DEFAULT_ENDPOINT = 'https://fe-26.qas.bing.net/sdf/'
DEFAULT_SCOPES = ['https://substrate.office.com/llmapi/LLMAPI.dev']
DEFAULT_API = 'chat/completions'

class LLMClient:
    _ENDPOINT = DEFAULT_ENDPOINT
    _SCOPES = DEFAULT_SCOPES
    _API = DEFAULT_API

    def __init__(self, scenario_id: str = None):
        self._scenario_id = scenario_id or "fd004048-ba97-46c8-9b09-6f566bdcd2d7"
        self._cache = SerializableTokenCache()
        self._cache_path = ".llmapi-dev.bin"
        self._app = PublicClientApplication(
            "545f9f54-6dca-4fce-9c2a-abd65266524f",
            authority="https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47",
            token_cache=self._cache,
        )
        self._token_lock = threading.Lock()
        if os.path.exists(self._cache_path):
            self._cache.deserialize(open(self._cache_path, "r").read())

        self.token = None
        atexit.register(self._save_token)
    
    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}. Response: {response.text}")
        return ChatCompletion(**response.json())
    
    def _get_token(self):
        with self._token_lock:
            # Use cached token if available
            if self.token:
                return self.token
            accounts = self._app.get_accounts()
            result = None
            if accounts:
                # Assuming the end user chose this one
                chosen = accounts[0]
                # Now let's try to find a token in cache for this account
                result = self._app.acquire_token_silent(self._SCOPES, account=chosen)
            if not result:
                # So no suitable token exists in cache. Let's get a new one from AAD.
                flow = self._app.initiate_device_flow(scopes=self._SCOPES)
                if "user_code" not in flow:
                    raise ValueError(
                        "Fail to create device flow. Err: %s"
                        % json.dumps(flow, indent=4)
                    )
                print(flow["message"])
                result = self._app.acquire_token_by_device_flow(flow)
                self._save_token()
            self.token = result.get("access_token", None)
            return self.token

    def _save_token(self):
        if self._cache.has_state_changed:
            with open(self._cache_path, "w") as f:
                f.write(self._cache.serialize())