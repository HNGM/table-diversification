import sys
sys.path.append(".")
import argparse
from src.utils.utils import ROOT_DIR, read_json, write_json
from pathlib import Path
import datetime
import re
from typing import List, Any
from research.evaluation.info import Info
from research.evaluation.evaluate import evaluate
from prose.llm import ChatModel, ChatRequest, Message, Role, SubstrateClient, SubstrateModels, Function
from prose.llm.agent import Agent, Tool, Environment, ToolResult
from prose.llm.model import StringProperty
from prose.llm.models import ModelSpecification, ModelSupports
from research.evaluation.utils import get_prompt
from research.agents.output_format import get_response_format, get_python_response_format
from research.agents.utils.code_tool import CodeTool
from src.utils.data_preview import get_data_preview_markdown
from tqdm import tqdm
from research.agents.utils.model_response import JsonResponseParser
import traceback
from research.evaluation.stats import get_tool_call_count_completion_prosellm
import json
from research.evaluation.utils import fix_json_serialization

from research.evaluation.prose_llm_main import load_data

model_name_ans_extract = ModelSpecification("dev-gpt-5-reasoning", ModelSupports.Chat | ModelSupports.Completion)
ans_extract_model = ChatModel(model_name_ans_extract, SubstrateClient(), suppress=True)

AnsExtractorPrompt = """
You are given the output of a python script execution. Your task is to extract the final answer from the output in the specified JSON format.
Question: {user_question}
Python Script Output:
{code_output}
"""

COMPLETION_MODELS = ["dev-mistral-7b-instruct-v02"]
ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "overall_distorted_dataset"
# ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "27122025_distorted_dataset"
DATA_MODE = "disturbed"
MODEL = "dev-deepseek-r1-distill-qwen-32b"
PROMPT_MODE = "default_mistake_tool_calling"
INGEST_MODE = "markdown"

def chat_model_toolcalling_workflow(model: ChatModel, info: Info, config) -> dict:
    eval_result = []
    for _ in range(config.pass_rate):
        code_tool = CodeTool()
        code_tool.upload_files([info.data_file])
        messages = [
            Message(role=Role.System, content=get_prompt(PROMPT_MODE) + "\n" + get_python_response_format())
        ]
        if config.ingest_mode == "markdown":
            markdown = get_data_preview_markdown(info.data_file)
            messages.append(Message(role=Role.User, content=f"The data is as follows:\n{markdown}\nIt has also been uploaded to the sandbox by the name '{info.data_file.name}'.\n\n{info.query}"))
        else:
            messages.append(Message(role=Role.User, content=f"Given the image of the table, answer the following question:\n{info.query}. The same data has been uploaded to the sandbox by the name '{info.data_file.name}'.", image=ROOT_DIR / info.image_file))
        max_iterations = 5
        eval = None
        for _ in range(max_iterations):
            response = model.chat(
                messages, 
                ChatRequest(max_completion_tokens=2048, n=1, temperature=0.0)
            )
            result_message = response.text
            messages.append(Message(role=Role.Assistant, content=result_message))
            
            python_pattern = r'```python\s*(.*?)\s*```'
            python_matches = re.findall(python_pattern, result_message, re.DOTALL)

            if python_matches:
                execution_results = []
                success_found = False
                for idx, code_block in enumerate(python_matches, start=1):
                    code_response = code_tool.run_code(code_block)
                    if code_response.exit_code == 0:
                        result_text = f"Code block {idx} executed successfully:\n{code_response.log}"
                        execution_results.append(result_text)
                        # Use answer extraction model to parse the output
                        try:
                            ans_ext_output = ans_extract_model.chat([Message(role=Role.User, content=AnsExtractorPrompt.replace("{user_question}", info.query).replace("{code_output}", code_response.log)+get_response_format())])
                            ans_ext_resp = ans_ext_output.text
                            agent_output = JsonResponseParser._parse_raw_response(ans_ext_resp)
                            agent_output = fix_json_serialization(agent_output)
                            eval = evaluate(
                                gt_answer=info.answer,
                                gt_dtype=info.dtype,
                                pred_answer=agent_output['answer'],
                                pred_dtype=agent_output['dtype']
                            )
                            eval_result.append({
                                'agent_response': agent_output,
                                'raw_response': [msg.model_dump(mode="json") for msg in messages],
                                'eval': eval,
                                "tools": len([msg.content for msg in messages if msg.role == Role.Assistant and "```python" in msg.content])
                            })
                            success_found = True
                            break
                        except ValueError as e:
                            print(f"Error in answer extraction: {e}")
                            result_text = f"Code block {idx} executed successfully:\n{code_response.log}" + get_response_format()
                            execution_results.append(result_text)
                    else:
                        result_text = f"Code block {idx} failed (exit code {code_response.exit_code}):\n{code_response.log}"
                        execution_results.append(result_text)
                if success_found:
                    break
                messages.append(Message(role=Role.User, content="Sandbox execution results:\n" + "\n".join(execution_results)))
                continue
            
            try:
                agent_output = JsonResponseParser._parse_raw_response(result_message)
                agent_output = fix_json_serialization(agent_output)
            except:
                messages.append(Message(role=Role.User, content="The previous response did not contain valid Python code or a final JSON answer. Please try again."))
                continue
            eval = evaluate(
                gt_answer=info.answer,
                gt_dtype=info.dtype,
                pred_answer=agent_output['answer'],
                pred_dtype=agent_output['dtype']
            )
            eval_result.append({
                'agent_response': agent_output,
                'raw_response': [msg.model_dump(mode="json") for msg in messages],
                'eval': eval,
                "tools": len([msg.content for msg in messages if msg.role == Role.Assistant and "```python" in msg.content])
            })
            break
        if eval is None:
            eval_result.append({
                'agent_response': {"answer": None, "dtype": None},
                'raw_response': [msg.model_dump(mode="json") for msg in messages],
                'eval': False,
                "tools": len([msg.content for msg in messages if msg.role == Role.Assistant and "```python" in msg.content])
            })
    info_dict = info.model_dump(mode="json")
    info_dict['eval'] = eval_result
    return info_dict

def get_config(args):
    # Define parser
    parser = argparse.ArgumentParser(description='Run evaluation on model')
    parser.add_argument('--input-file', default= ARTIFACT_DIR_SOURCE / f"{DATA_MODE}.json")
    parser.add_argument('--output-file', default=ROOT_DIR / "research" / "results" / datetime.datetime.now().strftime('%d%m%y') / f"v2_{ARTIFACT_DIR_SOURCE.name}_{DATA_MODE}_{PROMPT_MODE}_{INGEST_MODE}_{MODEL}.json")
    parser.add_argument('--nproc', type=int, default=1, help='Number of parallel processes')
    parser.add_argument('--model', type=str, default=MODEL, help='model to run the process on')
    parser.add_argument('--resume', action="store_true", help='Resume from the last checkpoint')
    parser.add_argument('--pass-rate', type=int, default=3, help='Set the pass@k rate for evaluation')
    parser.add_argument('--ingest-mode', type=str, default=INGEST_MODE, help='Ingest mode for data processing')
    config = parser.parse_args(args)

    config.input_file = Path(config.input_file)
    config.output_file = Path(config.output_file)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)

    return config 

def main(args):
    config = get_config(args)
    if config.output_file.exists() and config.resume:
        print(f"Output file {config.output_file} already exists. Will resume from the last checkpoint.")
        existing_results = read_json(config.output_file)
        processed_indices = {item['index'] for item in existing_results}
        print(f"Found {len(processed_indices)} already processed items.")
    else:
        processed_indices = set()
        existing_results = []
    info_list = load_data(config)
    rem_info_list = [inf for inf in info_list if inf.index not in processed_indices]
    model_name = ModelSpecification(config.model, ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)

    print(f"Resuming evaluation. {len(rem_info_list)} out of {len(info_list)} remaining.")
    
    # Process each info with fresh environment and agent per run
    for info in tqdm(rem_info_list):
        try:
            result = chat_model_toolcalling_workflow(model, info, config)
        except Exception as e:
            print(f"Error processing index {info.index}: {str(e)}")
            print(traceback.format_exc())
            continue
        
        # Validate result is JSON serializable before adding
        try:
            # Test serialization
            json.dumps(result)
            existing_results.append(result)
            write_json(existing_results, config.output_file)
        except (TypeError, ValueError) as e:
            print(f"Error serializing result for index {info.index}: {str(e)}")
            print(f"Skipping corrupted result for index {info.index}")
            continue
        
    




if __name__ == "__main__":
    main(sys.argv[1:])
