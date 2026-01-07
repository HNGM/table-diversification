import sys
sys.path.append(".")
from src.interfaces.workflow import Workflow
from src.interfaces import UserMessage, Message
from research.evaluation.info import Info
from src.utils.llm_config import LLMConfig
import argparse
from pathlib import Path
from src.utils.utils import ROOT_DIR, write_json
import datetime
import sys
from research.agents.sandbox_agent import FunctionCallAdaAgent
from research.agents.no_function_call_agent import NoFunctionCallAdaAgent
from research.evaluation.utils import get_prompt
from typing import List, Union, Type
from research.agents.output_format import get_response_format
from research.evaluation.evaluate import evaluate
from src.utils.data_preview import get_data_preview_markdown

# Agent framework mapping
AGENT_FRAMEWORKS = {
    "FunctionCallAdaAgent": FunctionCallAdaAgent,
    "NoFunctionCallAdaAgent": NoFunctionCallAdaAgent
}

ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "19122025_processed_dataset"
DATA_MODE = "disturbed"
MODEL = "dev-deepseek-r1-full"
PROMPT_MODE = "default_mistake_no_sandbox"
INGEST_MODE = "markdown"
FRAMEWORK = "NoFunctionCallAdaAgent"

class EvaluationWorkflow(Workflow):
    def __init__(
        self,
        llm_config_path: Union[Path, str],
        input_file: Union[Path, str],
        output_file: Union[Path, str],
        model: str,
        name: str = "EvaluationWorkflow",
        nproc: int = 1,
        resume: bool = False,
        pass_rate: int = 3,
        ingest_mode: str = "none",
        framework: Union[str, Type] = "FunctionCallAdaAgent"
    ):
        super().__init__(
            llm_config_path=llm_config_path,
            input_file=input_file,
            output_file=output_file,
            model=model,
            name=name,
            nproc=nproc,
            resume=resume
        )
        self.pass_rate = pass_rate
        self.ingest_mode = ingest_mode
        
        # Convert string framework name to class
        if isinstance(framework, str):
            if framework not in AGENT_FRAMEWORKS:
                raise ValueError(f"Unknown framework: {framework}. Available: {list(AGENT_FRAMEWORKS.keys())}")
            self.framework = AGENT_FRAMEWORKS[framework]
        else:
            self.framework = framework


    def workflow(self, llm_config: LLMConfig, info: Info) -> dict:
        eval_result = []
        # Assuming that each query is based on a single file
        for _ in range(self.pass_rate):
            agent = self.framework(llm_config, prompt=get_prompt(PROMPT_MODE) + "\n" + get_response_format())
            msgs: List[Message] = []
            if self.framework not in [NoFunctionCallAdaAgent]:
                msgs.append(agent.upload_files(files=[info.data_file], metadata=f"Answer the user's query based on the uploaded file name: {info.data_file.name}"))
            if self.ingest_mode == "screenshot":
                msgs.append(agent.upload_image_files(image_files=[info.image_file] if info.image_file else [], metadata=f"Answer the user's query based on the uploaded image of the data."))
            elif self.ingest_mode == "markdown":
                markdown = get_data_preview_markdown(info.data_file)
                msgs.append(UserMessage(content=f"The data preview is as follows:\n{markdown}"))
            msgs.append(UserMessage(content=info.query))
            agent_output = agent.run(msgs)
            agent_response = agent_output.ParsedResponse
            eval = evaluate(
                gt_answer=info.answer,
                gt_dtype=info.dtype,
                pred_answer=agent_response['answer'],
                pred_dtype=agent_response['dtype']
            )
            eval_result.append({
                'agent_response': agent_response,
                'raw_response': agent_output.RawResponse,
                'eval': eval
            })
        info_dict = info.model_dump(mode="json")
        info_dict['eval'] = eval_result
        return info_dict
    
    def load_data(self) -> List['Info']:
        print(f"Using benchmark file: {self.input_file}")
        benchmark_info = Info.get_info(self.input_file)
        print(f"Got {len(benchmark_info)} benchmark infos")
        return benchmark_info

def get_config(args):
    # Define parser
    parser = argparse.ArgumentParser(description='Run evaluation on model')
    parser.add_argument('--input-file', default= ARTIFACT_DIR_SOURCE / f"{DATA_MODE}.json")
    parser.add_argument('--output-file', default=ROOT_DIR / "research" / "results" / datetime.datetime.now().strftime('%d%m%y') / f"{DATA_MODE}_{PROMPT_MODE}_{INGEST_MODE}_{MODEL}.json")
    parser.add_argument('--llm-config-path', default=ROOT_DIR / "config" / "default_llm_config.json", help='Path to the LLM config file for running user-proxy')
    parser.add_argument('--nproc', type=int, default=1, help='Number of parallel processes')
    parser.add_argument('--model', type=str, default=MODEL, help='model to run the process on')
    parser.add_argument('--resume', action="store_true", help='Resume from the last checkpoint')
    parser.add_argument('--pass-rate', type=int, default=3, help='Set the pass@k rate for evaluation')
    parser.add_argument('--ingest-mode', type=str, default=INGEST_MODE, help='Ingest mode for data processing')
    parser.add_argument('--framework', type=str, default=FRAMEWORK, help='Framework to use for the agent')
    config = parser.parse_args(args)

    config.input_file = Path(config.input_file)
    config.output_file = Path(config.output_file)
    config.llm_config_path = Path(config.llm_config_path)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)

    return config 

def main(args):
    config = get_config(args)
    workflow = EvaluationWorkflow(**vars(config))
    workflow.run()

if __name__ == "__main__":
    main(sys.argv[1:])