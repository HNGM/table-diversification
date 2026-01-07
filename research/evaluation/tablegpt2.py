import sys
sys.path.append(".")
from research.agents.no_function_call_agent import NoFunctionCallAdaAgent
from src.utils.llm_config import load_llm_configs, LLMConfig
from src.utils.utils import ROOT_DIR, read_json, write_json
from src.utils.data_preview import get_data_preview_markdown
from src.interfaces import UserMessage, Message, AssistantMessage
from research.agents.instr_chat_agent import InstrChatAgent
from research.agents.utils.code_tool import CodeTool
from research.agents.utils.model_response import JsonResponseParser
from research.evaluation.evaluate import evaluate
import datetime
from tqdm import tqdm
from pathlib import Path
import traceback
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from research.agents.output_format import get_response_format, get_python_response_format
from llama_cpp import Llama
from research.evaluation.utils import get_prompt
from prose.llm import ChatModel, ChatRequest, Message as proseMessage, Role, SubstrateClient, SubstrateModels, Function
from prose.llm.agent import Agent, Tool, Environment, ToolResult
from prose.llm.model import StringProperty
from prose.llm.models import ModelSpecification, ModelSupports
from research.evaluation.utils import fix_json_serialization

model_name = ModelSpecification("dev-gpt-5-chat", ModelSupports.Chat | ModelSupports.Completion)
ans_extract_model = ChatModel(model_name, SubstrateClient(), suppress=True)

DEFAULT_PROMPT = """
DATA CONTEXT:
{data_preview}

Question: {user_question}

**Output format:**
Your output response should be provided in the following format:

```json
{
    "answer": Union[str, int, float, list, pd.Series, dict, set] <Mention the answer for the query>
    "dtype": str <Mention the data type of the answer you provided. It must belong to one of the options listed above.>
}
```

## Instructions
1) You must provide your answer in one of the following data types: "int", "float", "str", "list", "dict", "set" or "pd.Series".
# Rules on choosing the required dtype:
- If your answer needs to preserve order with only the relevant items, mention them as a list.
- If your answer does not have to preserve order with only the relevant items, mention them as a set.
- If your answer contains items along with their values where the order needs to be preserved, write the answer as a dictionary and mention the dtype as "pd.Series".
- If your answer contains items along with their values where the order need not be preserved, mention them as "dict".
2) In case the answer is in percentage, mention only the figure along with its datatype without the `%` sign.
3) You should not generate a python script as part of your answer. Always return the answer in the specified json format.
"""

TOOL_PROMPT = """
You are required to answer the user's question based on the provided data context. You have access to a Python coding tool that you can use to analyze the data and derive your answer. You should first generate the correct python script to obtain the solution and then you would receive instruction on how to present the answer.

DATA CONTEXT:
{data_preview}

Question: {user_question}

Generate a python script to solve the above in the following format:
```python
<python code>
```

- Do not include placeholders in your code. Use the actual file name provided in the data context.
"""

def workflow(llm_config: LLMConfig, info: dict):
    eval_result = []
    # for _ in range(3):
    markdown = get_data_preview_markdown(ROOT_DIR / info['data_file'])
    agent = InstrChatAgent(llm_config, prompt="Assist the user in answering data analysis questions based on the provided data context.")
    agent_output = agent.run([UserMessage(content=DEFAULT_PROMPT.replace("{data_preview}", markdown).replace("{user_question}", info['query']))])
    try:
        parsed_agent_resp = JsonResponseParser._parse_raw_response(agent_output.RawResponse)
        parsed_agent_resp = fix_json_serialization(parsed_agent_resp)
    except Exception as e:
        print(f"Error parsing agent response for index {info['index']}: {e}")
        parsed_agent_resp = {"answer": None, "dtype": None}
    print(agent_output.RawResponse)
    eval = evaluate(
        gt_answer=info['answer'],
        gt_dtype=info['dtype'],
        pred_answer=parsed_agent_resp['answer'],
        pred_dtype=parsed_agent_resp['dtype']
    )
    eval_result.append({
        'agent_response': parsed_agent_resp,
        'raw_response': agent_output.RawResponse,
        'eval': eval
    })
    info['eval'] = eval_result
    return info

def tool_workflow(llm_config: LLMConfig, info: dict):
    markdown = get_data_preview_markdown(ROOT_DIR / info['data_file'])
    code_tool = CodeTool()
    code_tool.upload_files([ROOT_DIR / info['data_file']])
    agent = InstrChatAgent(llm_config, prompt=TOOL_PROMPT.replace("{data_preview}", markdown).replace("{user_question}", info['query']))
    messages = [UserMessage(content=f"Use the uploaded file to answer the question. The uploaded file is located at: {info['data_file']}")]
    for _ in range(5):
        agent_output = agent.run(messages)
        python_pattern = r'```python\s*(.*?)\s*```'
        python_matches = re.findall(python_pattern, agent_output.RawResponse, re.DOTALL)
        messages.append(AssistantMessage(content=agent_output.RawResponse))
        if python_matches:
            execution_results = []
            for idx, code_block in enumerate(python_matches, start=1):
                code_response = code_tool.run_code(code_block)
                if code_response.exit_code == 0:
                    result_text = f"Code block {idx} executed successfully:\n{code_response.log}" + get_response_format()
                else:
                    result_text = f"Code block {idx} failed (exit code {code_response.exit_code}):\n{code_response.log}"
                execution_results.append(result_text)
            combined_response = "\n\n".join(execution_results)
            messages.append(Message(role="user", content=combined_response))
            continue
        try:
            parsed_agent_resp = JsonResponseParser._parse_raw_response(agent_output.RawResponse)
        except:
            messages.append(Message(role="user", content="The response format is incorrect. Please provide the answer in the specified JSON format.")) 
            parsed_agent_resp = {"answer": None, "dtype": None}
            continue
        eval = evaluate(
            gt_answer=info['answer'],
            gt_dtype=info['dtype'],
            pred_answer=parsed_agent_resp['answer'],
            pred_dtype=parsed_agent_resp['dtype']
        )
        break
    info['eval'] = [{
        'agent_response': parsed_agent_resp,
        'raw_response': [msg.to_openapi_format() for msg in messages],
        'eval': eval,
        'tools': len([msg for msg in messages if isinstance(msg, AssistantMessage) and "```python" in msg.content])
    }]
    return info


    
def llmcpp_toolcalling_workflow(llm, info: dict):
    markdown = get_data_preview_markdown(ROOT_DIR / info['data_file'])
    code_tool = CodeTool()
    code_tool.upload_files([ROOT_DIR / info['data_file']])
    messages = [Message(role="system", content=get_prompt("default_tool_calling") + "\n" + get_python_response_format())]
    messages.append(UserMessage(content=f"DATA CONTEXT:\n{markdown}\n\nQuestion: {info['query']}\n\nUse the uploaded file name {Path(info['data_file']).name} in your python script. Do not use a placeholder."))
    parsed_agent_resp = {"answer": None, "dtype": None}
    eval = None
    for _ in range(5):
        response = llm.create_chat_completion(
            messages=[msg.to_openapi_format() for msg in messages]
        )
        messages.append(Message(role="assistant", content=response['choices'][0]['message']['content']))
        python_pattern = r'```python\s*(.*?)\s*```'
        python_matches = re.findall(python_pattern, response['choices'][0]['message']['content'], re.DOTALL)
        if python_matches:
            execution_results = []
            for idx, code_block in enumerate(python_matches, start=1):
                code_response = code_tool.run_code(code_block)
                if code_response.exit_code == 0:
                    result_text = f"Code block {idx} executed successfully:\n{code_response.log}" + get_response_format()
                else:
                    result_text = f"Code block {idx} failed (exit code {code_response.exit_code}):\n{code_response.log}\n please fix this issue and provide the corrected code."
                execution_results.append(result_text)
            combined_response = "\n\n".join(execution_results)
            messages.append(Message(role="user", content=combined_response))
            continue
        try:
            parsed_agent_resp = JsonResponseParser._parse_raw_response(response['choices'][0]['message']['content'])
            parsed_agent_resp = fix_json_serialization(parsed_agent_resp)
        except:
            messages.append(Message(role="user", content="Either you did not generate python code or your response format is incorrect. Please fix them based on the rules defined above.")) 
            continue
        eval = evaluate(
            gt_answer=info['answer'],
            gt_dtype=info['dtype'],
            pred_answer=parsed_agent_resp['answer'],
            pred_dtype=parsed_agent_resp['dtype']
        )
        break
    if eval is None:
        info['eval'] = [{
            'agent_response': {"answer": None, "dtype": None},
            'raw_response': [msg.to_openapi_format() for msg in messages],
            'eval': False,
            "tools": len([msg for msg in messages if msg.role == "assistant" and "```python" in msg.content])
        }]
    else:
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
        n_ctx=4096,  # Increase context window to handle longer prompts
        n_batch=512,  # Batch size for prompt processing
    )
    dataset = read_json(ROOT_DIR / "research" / "dataset" / "19122025_processed_dataset" / "original.json")
    output_file = ROOT_DIR / r"research\results\281225\original_default_tool_calling_markdown_tablegpt2-7b.json"
    output_dataset = read_json(output_file) if output_file.exists() else []
    for data in tqdm(dataset):
        if data['index'] in set([item['index'] for item in output_dataset]):
            print(f"Skipping already processed index: {data['index']}")
            continue
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
