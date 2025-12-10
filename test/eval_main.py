import sys
sys.path.append(".")
from src.utils.utils import read_json, ROOT_DIR
from src.utils.llm_config import LLMConfig, load_llm_configs
from research.agents.agent import FunctionCallAdaAgent
from research.evaluation.utils import get_prompt
from research.agents.output_format import get_response_format
from src.interfaces import UserMessage, Message
from research.evaluation.info import Info




def test_eval_workflow(info):
    llm_config = load_llm_configs(ROOT_DIR / "config" / "default_llm_config.json", "dev-gpt-41-shortco-2025-04-14")[0]
    agent = FunctionCallAdaAgent(llm_config, prompt=get_prompt("default") + "\n" + get_response_format())
    # Assuming that each query is based on a single file
    upload_file_msg = agent.upload_files(files=[info.data_file], metadata=f"Answer the user's query based on the uploaded file name: {info.data_file.name}")
    agent_output = agent.run([upload_file_msg, UserMessage(content=info.query + "\nCan you also say if you can read the data and if so provide the first 5 rows as mrkdown.")])
    print(agent_output.RawResponse)

def test():
    dataset = read_json("research/dataset/processed_dataset/original.json")
    index = "Clinic_Visits_original_2"
    data = [d for d in dataset if d["index"] == index][0]
    info = Info(**data)
    test_eval_workflow(info)





if __name__ == "__main__":
    test()