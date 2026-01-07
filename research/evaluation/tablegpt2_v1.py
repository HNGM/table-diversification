import sys
sys.path.append(".")
from src.utils.utils import ROOT_DIR, read_json, write_json
from src.utils.data_preview import get_data_preview_markdown
from src.interfaces import UserMessage, Message, AssistantMessage
from research.agents.utils.code_tool import CodeTool
from research.agents.utils.model_response import JsonResponseParser
from research.evaluation.evaluate import evaluate
import datetime
from tqdm import tqdm
from pathlib import Path
import traceback
import re
from research.agents.output_format import get_response_format, get_python_response_format
from llama_cpp import Llama
from research.evaluation.utils import get_prompt
from prose.llm import ChatModel, Message as proseMessage, Role, SubstrateClient
from prose.llm.models import ModelSpecification, ModelSupports
from research.evaluation.utils import fix_json_serialization

model_name = ModelSpecification("dev-gpt-5-reasoning", ModelSupports.Chat | ModelSupports.Completion)
ans_extract_model = ChatModel(model_name, SubstrateClient(), suppress=True)

DATASET_MODE = "disturbed"
PROMPT_MODE = "mistake"

ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "overall_distorted_dataset"
# ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "27122025_distorted_dataset"

MISTAKE_PROMPT = "If you encounter scnearios where the table seems incorrect either structurally or semantically (e.g., shifted rows, shifted columns, semantic misalignment, etc.), ensure that your code accounts for these inconsistencies to provide an accurate and robust answer."

DEFAULT_PROMPT = """
You are provided with a data context and a question related to that data. Your task is to analyze the data and provide a python script that answers the question based on the data context. You should not use placeholders in your code. Use the actual file name provided in the data context and the python code should be complete and run end to end without errors. In case there are errors you will be provided with the error logs and you should fix the code accordingly.

{mistake_instr}

Generate a python script to solve the above in the following format:
```python
<python code>
```

DATA CONTEXT:
{data_preview}

filename to use in the code: {data_file}

Question: {user_question}
"""

AnsExtractorPrompt = """
You are given the output of a python script execution. Your task is to extract the final answer from the output in the specified JSON format.
Question: {user_question}
Python Script Output:
{code_output}
"""
    
def llmcpp_toolcalling_workflow(llm, info: dict):
    markdown = get_data_preview_markdown(ROOT_DIR / info['data_file'])
    code_tool = CodeTool()
    code_tool.upload_files([ROOT_DIR / info['data_file']])
    messages = [Message(role="system", content=DEFAULT_PROMPT.replace("{data_preview}", markdown).replace("{user_question}", info['query']).replace("{data_file}", Path(info['data_file']).name).replace("{mistake_instr}", MISTAKE_PROMPT if PROMPT_MODE == "mistake" else ""))]
    parsed_agent_resp = {"answer": None, "dtype": None}
    ans_ext_resp = None
    for _ in range(5):
        response = llm.create_chat_completion(
            messages=[msg.to_openapi_format() for msg in messages]
        )
        messages.append(Message(role="assistant", content=response['choices'][0]['message']['content']))
        python_pattern = r'```python\s*(.*?)\s*```'
        python_matches = re.findall(python_pattern, response['choices'][0]['message']['content'], re.DOTALL)
        if python_matches:
            code_block = python_matches[0]
            code_response = code_tool.run_code(code_block)
            
            if code_response.exit_code == 0:
                try:
                    ans_ext_output = ans_extract_model.chat([proseMessage(role=Role.User, content=AnsExtractorPrompt.replace("{user_question}", info['query']).replace("{code_output}", code_response.log)+get_response_format())])
                except ValueError as e:
                    print(f"Error in answer extraction: {e}")
                    break
                ans_ext_resp = ans_ext_output.text 
                result_text = f"Code block executed successfully:\n{code_response.log}"
                messages.append(Message(role="user", content=result_text))               
                break
            else:
                result_text = f"Code block failed (exit code {code_response.exit_code}):\n{code_response.log}\n please fix this issue and provide the corrected code."
                messages.append(Message(role="user", content=result_text))
                continue
                
    if ans_ext_resp is None:
        info['eval'] = [{
            'agent_response': {"answer": None, "dtype": None},
            'raw_response': [msg.to_openapi_format() for msg in messages],
            'eval': False,
            "tools": len([msg for msg in messages if msg.role == "assistant" and "```python" in msg.content])
        }]
    else:
        parsed_agent_resp = JsonResponseParser._parse_raw_response(ans_ext_resp)
        parsed_agent_resp = fix_json_serialization(parsed_agent_resp)
        eval = evaluate(
            gt_answer=info['answer'],
            gt_dtype=info['dtype'],
            pred_answer=parsed_agent_resp['answer'],
            pred_dtype=parsed_agent_resp['dtype']
        )
        info['eval'] = [{
            'agent_response': parsed_agent_resp,
            'raw_response': [msg.to_openapi_format() for msg in messages],
            'eval': eval,
            'tools': len([msg for msg in messages if msg.role == "assistant" and "```python" in msg.content])
        }]
    return info


    


def main():
    # llm_config = load_llm_configs(ROOT_DIR / "config" / "default_llm_config.json", "tablegpt2-7b-local")[0]
    llm = Llama.from_pretrained(
        repo_id = "QuantFactory/TableGPT2-7B-GGUF",
        filename="TableGPT2-7B.Q2_K.gguf",
        n_ctx=8192,  # Increase context window to handle longer prompts
        n_batch=512,  # Batch size for prompt processing
        verbose=False,  # Suppress meta information and loading messages
    )
    dataset = read_json(ARTIFACT_DIR_SOURCE / f"{DATASET_MODE}.json")
    output_file = ROOT_DIR / "research" / "results" / datetime.datetime.now().strftime('%d%m%y') / f"{ARTIFACT_DIR_SOURCE.name}_{DATASET_MODE}_{PROMPT_MODE}_tablegpt2-7b.json"
    output_dataset = read_json(output_file) if output_file.exists() else []
    processed_indices = set([item['index'] for item in output_dataset])
    rem_dataset = [d for d in dataset if d['index'] not in processed_indices]
    print(f"Starting evaluation on {len(dataset)} samples, already completed: {len(output_dataset)}")
    for data in tqdm(rem_dataset):
        try:
            result = llmcpp_toolcalling_workflow(llm, data)
        except Exception as e:
            print(f"Error processing index {data['index']}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue
        output_dataset.append(result)
        write_json(output_dataset, output_file)

if __name__ == "__main__":
    main()
