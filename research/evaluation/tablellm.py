import sys
import os
sys.path.append(".")
import argparse
import json
import signal
import atexit
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

ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "wikitq_dataset_filtered"
DATASET_MODE = "original"
PROMPT_MODE = "default"
TOOLING = True

# ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "27122025_distorted_dataset"

MISTAKE_PROMPT = "If you encounter scnearios where the table seems incorrect either structurally or semantically (e.g., shifted rows, shifted columns, semantic misalignment, etc.), ensure that your analysis accounts for these inconsistencies to provide an accurate and robust answer."

TOOLING_PROMPT = """
[INST]Below is the table data. You need to write a Python program that reads the data as 'data.xlsx' to solve the problem.

{mistake_instr}

DATA CONTEXT:
{csv_data}

Question: {question}

Enclose the python code in a python code block as shown below:
```python
<python code>
```

- Use 'data.xlsx' as the file name in your code.
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
    messages = [Message(role="user", content=TOOLING_PROMPT.replace("{csv_data}", markdown).replace("{question}", info['query']).replace("{mistake_instr}", MISTAKE_PROMPT if PROMPT_MODE == "mistake" else ""))]
    parsed_agent_resp = {"answer": None, "dtype": None}
    ans_ext_resp = None
    for _ in range(3):
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
        data_file_name = Path(info['data_file']).name
        if code_block.count("'data.xlsx'") == 1:
            code_block = code_block.replace("'data.xlsx'", f"'{data_file_name}'")
        else:
            code_block = re.sub(
                r"pd\.read_excel\(\s*'data\.xlsx'\s*\)",
                f"pd.read_excel('{data_file_name}')",
                code_block,
            )
        code_response = code_tool.run_code(code_block)
        
        if code_response.exit_code == 0:
            ans_ext_output = ans_extract_model.chat([proseMessage(role=Role.User, content=AnsExtractorPrompt.replace("{user_question}", info['query']).replace("{code_output}", code_response.log)+get_response_format())])
            ans_ext_resp = ans_ext_output.text 
            result_text = f"Code block executed successfully:\n{code_response.log}"
            messages.append(UserMessage(content=result_text))               
            break
        else:
            result_text = f"Code block execution failed with exit code {code_response.exit_code}:\n{code_response.log}\nPlease fix the code and try again."
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
    

    


def _read_jsonl_lenient(jsonl_path: Path) -> list:
    """Read a .jsonl file, skipping blank/corrupt trailing lines."""
    results = []
    if not jsonl_path.exists():
        return results
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_no} in {jsonl_path}: {e}")
                continue
    return results


def _dedupe_by_index(results: list) -> list:
    """Keep the last occurrence per `index` to avoid duplicates."""
    by_index = {}
    order = []
    for item in results:
        idx = item.get("index")
        if idx is None:
            order.append(id(item))
            by_index[id(item)] = item
            continue
        if idx not in by_index:
            order.append(idx)
        by_index[idx] = item
    return [by_index[k] for k in order]


def _finalize_jsonl_to_json(jsonl_path: Path, json_path: Path):
    """Consolidate the streaming .jsonl checkpoint into the .json output."""
    if not jsonl_path.exists():
        return
    results = _read_jsonl_lenient(jsonl_path)
    if not results and json_path.exists():
        print(f"No results in {jsonl_path}; leaving existing {json_path} untouched.")
        try:
            jsonl_path.unlink()
        except OSError as e:
            print(f"Warning: failed to delete {jsonl_path}: {e}")
        return
    results = _dedupe_by_index(results)
    try:
        write_json(results, json_path)
        print(f"Finalized {len(results)} results to {json_path}")
    except Exception as e:
        print(f"Failed to finalize jsonl -> json: {e}")
        return
    try:
        jsonl_path.unlink()
    except OSError as e:
        print(f"Warning: failed to delete {jsonl_path}: {e}")


def get_config(args):
    parser = argparse.ArgumentParser(description='Run TableLLM evaluation')
    default_output = ROOT_DIR / "research" / "results" / datetime.datetime.now().strftime('%d%m%y') / f"{ARTIFACT_DIR_SOURCE.name}_{DATASET_MODE}_{PROMPT_MODE}_{'markdown' if TOOLING else 'no_sandbox_markdown'}_tablellm.json"
    parser.add_argument('--input-file', default=ARTIFACT_DIR_SOURCE / f"{DATASET_MODE}.json")
    parser.add_argument('--output-file', default=default_output)
    parser.add_argument('--resume', action="store_true", help='Resume from the last checkpoint')
    config = parser.parse_args(args)
    config.input_file = Path(config.input_file)
    config.output_file = Path(config.output_file)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    return config


def main(args):
    config = get_config(args)

    llm = Llama.from_pretrained(
        repo_id="tensorblock/RUCKBReasoning_TableLLM-7b-GGUF",
        filename="TableLLM-7b-Q2_K.gguf",
        n_ctx=8192,
        n_batch=512,
        verbose=False,
    )
    dataset = read_json(config.input_file)

    jsonl_path = config.output_file.with_suffix(".jsonl")
    existing_results: list = []
    processed_indices: set = set()

    if config.resume:
        if jsonl_path.exists():
            print(f"Found leftover streaming checkpoint {jsonl_path}. Recovering it.")
            existing_results = _read_jsonl_lenient(jsonl_path)
            if config.output_file.exists():
                try:
                    finalized = read_json(config.output_file)
                    jsonl_indices = {it.get("index") for it in existing_results}
                    extra = [it for it in finalized if it.get("index") not in jsonl_indices]
                    if extra:
                        print(f"Merging {len(extra)} extra items from {config.output_file}.")
                        existing_results.extend(extra)
                except Exception as e:
                    print(f"Warning: could not merge {config.output_file}: {e}")
        elif config.output_file.exists():
            print(f"Output file {config.output_file} exists. Seeding jsonl checkpoint from it.")
            existing_results = read_json(config.output_file)
        else:
            print("Resume requested but no prior checkpoint or output file found. Starting fresh.")

        existing_results = _dedupe_by_index(existing_results)
        if existing_results:
            tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for item in existing_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            os.replace(tmp_path, jsonl_path)
        elif jsonl_path.exists():
            jsonl_path.unlink()

        processed_indices = {item['index'] for item in existing_results if 'index' in item}
        print(f"Found {len(processed_indices)} already processed items.")
    else:
        if jsonl_path.exists():
            print(f"Fresh run: removing stale checkpoint {jsonl_path}.")
            jsonl_path.unlink()

    rem_dataset = [d for d in dataset if d['index'] not in processed_indices]
    print(f"Resuming evaluation. {len(rem_dataset)} out of {len(dataset)} remaining.")

    _finalized = {"done": False}

    def _finalize_once():
        if _finalized["done"]:
            return
        _finalized["done"] = True
        _finalize_jsonl_to_json(jsonl_path, config.output_file)

    atexit.register(_finalize_once)

    def _signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Finalizing results...")
        _finalize_once()
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    try:
        with open(jsonl_path, "a", encoding="utf-8") as ckpt_f:
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

                try:
                    line = json.dumps(result, ensure_ascii=False)
                except (TypeError, ValueError) as e:
                    print(f"Error serializing result for index {data['index']}: {e}")
                    continue

                ckpt_f.write(line + "\n")
                ckpt_f.flush()
    finally:
        _finalize_once()


if __name__ == "__main__":
    main(sys.argv[1:])
