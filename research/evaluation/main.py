import sys
sys.path.append(".")
from src.interfaces.workflow import Workflow
from src.interfaces import UserMessage, Message
from research.evaluation.info import Info
from src.utils.llm_config import LLMConfig, load_llm_configs
import argparse
from pathlib import Path
from src.utils.utils import ROOT_DIR
import datetime
import sys
from research.agents.agent import FunctionCallAdaAgent
from research.evaluation.utils import get_prompt
from typing import List, Any
import pandas as pd
import traceback
from research.agents.output_format import get_response_format
from research.evaluation.evaluate import evaluate

ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "processed_dataset"



class EvaluationWorkflow(Workflow):
    def workflow(self, llm_config: LLMConfig, info: Info) -> dict:
        agent = FunctionCallAdaAgent(llm_config, prompt=get_prompt("default") + "\n" + get_response_format())
        # Assuming that each query is based on a single file
        upload_file_msg = agent.upload_files(files=[info.data_file], metadata=f"Answer the user's query based on the uploaded file name: {info.data_file.name}")
        agent_output = agent.run([upload_file_msg, UserMessage(content=info.query)])
        agent_response = agent_output.ParsedResponse
        eval = evaluate(
            gt_answer=info.answer,
            gt_dtype=info.dtype,
            pred_answer=agent_response['answer'],
            pred_dtype=agent_response['dtype']
        )
        info_dict = info.model_dump(mode="json")
        info_dict['agent_response'] = agent_response
        info_dict['raw_response'] = agent_output.RawResponse
        info_dict['evaluation'] = eval
        return info_dict
    
    def load_data(self) -> List['Info']:
        print(f"Using benchmark file: {self.input_file}")
        benchmark_info = Info.get_info(self.input_file)
        print(f"Got {len(benchmark_info)} benchmark infos")
        return benchmark_info

def get_config(args):
    # Define parser
    parser = argparse.ArgumentParser(description='Run evaluation on model')
    parser.add_argument('--input-file', default= ARTIFACT_DIR_SOURCE / "disturbed.json")
    parser.add_argument('--output-file', default=ROOT_DIR / "research" / "results" / datetime.datetime.now().strftime('%d%m%y') / f"disturbed.json")
    parser.add_argument('--llm-config-path', default=ROOT_DIR / "config" / "default_llm_config.json", help='Path to the LLM config file for running user-proxy')
    parser.add_argument('--nproc', type=int, default=1, help='Number of parallel processes')
    parser.add_argument('--model', type=str, default="dev-gpt-41-shortco-2025-04-14", help='model to run the process on')
    parser.add_argument('--resume', action="store_true", help='Resume from the last checkpoint')
    config = parser.parse_args(args)

    config.input_file = Path(config.input_file)
    config.output_file = Path(config.output_file)
    config.llm_config_path = Path(config.llm_config_path)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)

    return config 

def main(args):
    config = get_config(args)
    workflow = EvaluationWorkflow(
        llm_config_path=config.llm_config_path,
        input_file=config.input_file,
        output_file=config.output_file,
        model=config.model,
        name="EvaluationWorkflow",
        nproc=config.nproc,
        resume=config.resume,
    )
    workflow.run()

if __name__ == "__main__":
    main(sys.argv[1:])