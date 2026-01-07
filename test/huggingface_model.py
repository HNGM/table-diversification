import sys
sys.path.append(".")
from src.utils.llm_config import LLMConfig, load_llm_configs
from src.interfaces import ChatAgent, ChatAgentConfig, UserMessage, AgentResponse, Message

def test():
    llm_config = load_llm_configs("config/default_llm_config.json", "tablegpt2-7b-local")[0]
    agent = ChatAgent(
        llm_config,
        config=ChatAgentConfig(
            model="tablegpt2-7b-local",
            name="TestChatAgent",
            instructions="You are a helpful assistant.",
        ),
        response_type=AgentResponse
    )
    output = agent.run([Message(role="system", content="Help the user"), UserMessage(content="What is the capital of France?"), UserMessage(content="Share two fun facts about it.")])
    print(output.RawResponse)

    
    

if __name__ == "__main__":
    test()