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
from research.agents.output_format import get_response_format
from llama_cpp import Llama
from prose.llm import ChatModel, Message as proseMessage, Role, SubstrateClient
from prose.llm.models import ModelSpecification, ModelSupports
from research.evaluation.utils import fix_json_serialization

DATASET_MODE = "disturbed"
PROMPT_MODE = "mistake"
TOOLING = True

ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "overall_distorted_dataset"
# ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "27122025_distorted_dataset"

MISTAKE_PROMPT = "If you encounter scnearios where the table seems incorrect either structurally or semantically (e.g., shifted rows, shifted columns, semantic misalignment, etc.), ensure that your analysis accounts for these inconsistencies to provide an accurate and robust answer."

TOOLING_PROMPT = """
[INST]Below is the table data. You need to write a Python program that reads the data (`{data_file}`) to solve the provided question.

{mistake_instr}

DATA CONTEXT:
{csv_data}

Question: {question}

Enclose the python code in a python code block as shown below:
```python
<python code>
```

- You should not use placeholders like `data.csv` in your code. Use the actual file name provided in the data context for reading the data `{data_file}`.
- The python code should be complete and run end to end without errors.
- In case there are errors you will be provided with the error logs and you should fix the code accordingly.
[/INST]
"""

DIRECT_ANSWER_PROMPT = """
[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question].

{mistake_instr}

Do not write any code. Provide your final answer directly based on the data presented in the [Table].

{output_format}

### [Table]
```
{table_in_csv}
```

### [Question]
{question}

### [Solution][INST/]
```json
"""

AnsExtractorPrompt = """
You are given the output of a python script execution. Your task is to extract the final answer from the output in the specified JSON format.
Question: {user_question}
Python Script Output:
{code_output}
"""

model_name = ModelSpecification("dev-gpt-5-reasoning", ModelSupports.Chat | ModelSupports.Completion)
ans_extract_model = ChatModel(model_name, SubstrateClient(), suppress=True)

def workflow(llm, info: dict) -> dict:
    markdown = get_data_preview_markdown(ROOT_DIR / info['data_file'])
    messages = [UserMessage(content=DIRECT_ANSWER_PROMPT.replace("{table_in_csv}", markdown).replace("{question}", info['query']).replace("{output_format}", get_response_format()).replace("{mistake_instr}", MISTAKE_PROMPT if PROMPT_MODE == "mistake" else ""))]
    response = llm.create_chat_completion(
        messages = [msg.to_openapi_format() for msg in messages]
    )
    response_text = response['choices'][0]['message']['content']
    try:
        parsed_agent_resp = JsonResponseParser._parse_raw_response(response_text[:response_text.find("```")])
        parsed_agent_resp = fix_json_serialization(parsed_agent_resp)
    except Exception as e:
        parsed_agent_resp = {"answer": None, "dtype": None}
    eval = evaluate(
        gt_answer=info['answer'],
        gt_dtype=info['dtype'],
        pred_answer=parsed_agent_resp['answer'],
        pred_dtype=parsed_agent_resp['dtype']
    )
    info['eval'] = [{
        'agent_response': parsed_agent_resp,
        'raw_response': response['choices'][0]['message']['content'],
        'eval': eval,
        "tools": 0
    }]
    return info

def tooling_workflow(llm, info: dict) -> dict:
    markdown = get_data_preview_markdown(ROOT_DIR / info['data_file'])
    code_tool = CodeTool()
    code_tool.upload_files([ROOT_DIR / info['data_file']])
    messages = [Message(role="user", content=TOOLING_PROMPT.replace("{csv_data}", markdown).replace("{question}", info['query']).replace("{data_file}", Path(info['data_file']).name).replace("{mistake_instr}", MISTAKE_PROMPT if PROMPT_MODE == "mistake" else ""))]
    parsed_agent_resp = {"answer": None, "dtype": None}
    ans_ext_resp = None
    for _ in range(5):
        response = llm.create_chat_completion(
            messages = [msg.to_openapi_format() for msg in messages]
        )
        messages.append(AssistantMessage(content=response['choices'][0]['message']['content']))
        python_pattern = r'```python\s*(.*?)\s*```'
        python_matches = re.findall(python_pattern, response['choices'][0]['message']['content'], re.DOTALL)

        if python_matches:
            code_block = python_matches[0]
        else:
            code_block = response['choices'][0]['message']['content']
        code_response = code_tool.run_code(code_block)
        
        if code_response.exit_code == 0:
            ans_ext_output = ans_extract_model.chat([proseMessage(role=Role.User, content=AnsExtractorPrompt.replace("{user_question}", info['query']).replace("{code_output}", code_response.log)+get_response_format())])
            ans_ext_resp = ans_ext_output.text 
            result_text = f"Code block executed successfully:\n{code_response.log}"
            messages.append(UserMessage(content=result_text))               
            break
        else:
            result_text = f"Code block execution failed with exit code {code_response.exit_code}:\n{code_response.log}\nPlease fix the code and try again. Use the data file `{Path(info['data_file']).name}` for reading the data."
            messages.append(UserMessage(content=result_text))
            continue
    if ans_ext_resp is None:
        info['eval'] = [{
            'agent_response': {"answer": None, "dtype": None},
            'raw_response': [msg.to_openapi_format() for msg in messages],
            'eval': False,
            "tools": len([msg for msg in messages if msg.role == "assistant"])
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
            'tools': len([msg for msg in messages if msg.role == "assistant"])
        }]
    return info
    

    


def main():
    llm = Llama.from_pretrained(
        repo_id="tensorblock/RUCKBReasoning_TableLLM-7b-GGUF",
        filename="TableLLM-7b-Q2_K.gguf",
        n_ctx=8192,  # Increase context window size (default is 512)
        n_batch=512,  # Number of tokens to process in parallel
        verbose=False,  # Disable verbose logging
    )
    dataset = read_json(ARTIFACT_DIR_SOURCE / f"{DATASET_MODE}.json")
    # output_file = ROOT_DIR / "research" / "results" / datetime.datetime.now().strftime('%d%m%y') / f"{ARTIFACT_DIR_SOURCE.name}_{DATASET_MODE}_{PROMPT_MODE}_{'tooling' if TOOLING else 'no_tooling'}_tablellm.json"
    output_file = Path(r"C:\repo\table-diversification\research\results\311225\overall_distorted_dataset_disturbed_mistake_tooling_tablellm.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_dataset = read_json(output_file) if output_file.exists() else []

    processed_indices = set([item['index'] for item in output_dataset])
    print(f"Already processed {len(processed_indices)} items. Resuming...")
    rem_dataset = [d for d in dataset if d['index'] not in processed_indices]

    for data in tqdm(rem_dataset):
        try:
            if TOOLING:
                result = tooling_workflow(llm, data)
            else:
                result = workflow(llm, data)
        except Exception as e:
            print(f"Error processing index {data['index']}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue
        output_dataset.append(result)
        write_json(output_dataset, output_file)

if __name__ == "__main__":
    main()
