import sys
sys.path.append(".")
from src.utils.utils import ROOT_DIR
from src.utils.substrate import LLMClient

def test():
    client = LLMClient("https://fe-26.qas.bing.net/sdf/")
    req_args = {}
    req_args["model"] = "dev-phi-4"
    req_args["messages"] = [
        {
            "role": "user",
            "content": "Hello, can you help me?"
        }
    ]
    response = client.send_request(req_args["model"], req_args)
    print(response)









if __name__ == "__main__":
    test()
