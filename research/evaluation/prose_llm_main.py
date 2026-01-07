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

COMPLETION_MODELS = ["dev-mistral-7b-instruct-v02"]
ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "overall_distorted_dataset"
# ARTIFACT_DIR_SOURCE = ROOT_DIR / "research" / "dataset" / "27122025_distorted_dataset"
DATA_MODE = "disturbed"
MODEL = "dev-gpt-51-2025-11-13"
PROMPT_MODE = "default_mistake_no_sandbox"
INGEST_MODE = "markdown"

class CodeEnvironment(Environment):
    """Environment that holds a CodeTool instance for code execution."""
    def __init__(self, code_tool: CodeTool = None):
        super().__init__()
        self.code_tool = code_tool if code_tool else CodeTool()
        self.uploaded_files = []  # Track uploaded files
    
    def upload_file(self, file_path: Path) -> Path:
        """Upload a file to the code sandbox."""
        uploaded = self.code_tool.upload_files([file_path])
        if uploaded:
            self.uploaded_files.extend([f.name for f in uploaded])
            return uploaded[0]
        return None
    
def completion_model_toolcalling_workflow(model: ChatModel, info: Info, config) -> dict:
    eval_result = []
    for _ in range(config.pass_rate):
        code_tool = CodeTool()
        code_tool.upload_files([info.data_file])
        messages = [
            "<|im_start|>system\n" + get_prompt(PROMPT_MODE) + "\n" + get_python_response_format() + "<|im_end|>"
        ]
        if config.ingest_mode == "markdown":
            markdown = get_data_preview_markdown(info.data_file)
            messages.append(f"<|im_start|>user\nThe data is as follows:\n{markdown}\nIt has also been uploaded to the sandbox by the name '{info.data_file.name}'.\n\n{info.query}<|im_end|>")
        else:
            messages.append(f"<|im_start|>user\n{info.query}<|im_end|>")
        
        max_iterations = 5
        eval = None
        for _ in range(max_iterations):
            messages.append("<|im_start|>assistant")
            
            if config.ingest_mode == "screenshot" and info.image_file and info.image_file.exists():
                response = model.chat(
                    [Message(role=Role.User, content="\n".join(messages)+"\nThe image for the table has been provided for context.", image=info.image_file)], 
                    ChatRequest(max_completion_tokens=4096, n=1, temperature=0.0)
                )
            else:
                response = model.chat(
                    [Message(role=Role.User, content="\n".join(messages))], 
                    ChatRequest(max_completion_tokens=2048, n=1, temperature=0.0)
                )
            result_message = response.text
            messages[-1] = messages[-1] + "\n" + result_message + "<|im_end|>"
            
            python_pattern = r'```python\s*(.*?)\s*```'
            python_matches = re.findall(python_pattern, result_message, re.DOTALL)

            if python_matches:
                execution_results = []
                for idx, code_block in enumerate(python_matches, start=1):
                    code_response = code_tool.run_code(code_block)
                    if code_response.exit_code == 0:
                        result_text = f"Code block {idx} executed successfully:\n{code_response.log}" + get_response_format()
                    else:
                        result_text = f"Code block {idx} failed (exit code {code_response.exit_code}):\n{code_response.log}"
                    execution_results.append(result_text)
                messages.append(f"<|im_start|>user\nSandbox execution results:\n" + "\n".join(execution_results) + "<|im_end|>")
                continue
            
            try:
                agent_output = JsonResponseParser._parse_raw_response(result_message)
                agent_output = fix_json_serialization(agent_output)
            except:
                messages.append(f"<|im_start|>user\nThe previous response did not contain valid Python code or a final JSON answer. Please try again.<|im_end|>")
                continue
            eval = evaluate(
                gt_answer=info.answer,
                gt_dtype=info.dtype,
                pred_answer=agent_output['answer'],
                pred_dtype=agent_output['dtype']
            )
            eval_result.append({
                'agent_response': agent_output,
                'raw_response': messages,
                'eval': eval,
                "tools": get_tool_call_count_completion_prosellm(messages)
            })
            break
        if eval is None:
            eval_result.append({
                'agent_response': {"answer": None, "dtype": None},
                'raw_response': messages,
                'eval': False,
                "tools": get_tool_call_count_completion_prosellm(messages)
            })
            
    info_dict = info.model_dump(mode="json")
    info_dict['eval'] = eval_result
    return info_dict

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
                for idx, code_block in enumerate(python_matches, start=1):
                    code_response = code_tool.run_code(code_block)
                    if code_response.exit_code == 0:
                        result_text = f"Code block {idx} executed successfully:\n{code_response.log}" + get_response_format()
                    else:
                        result_text = f"Code block {idx} failed (exit code {code_response.exit_code}):\n{code_response.log}"
                    execution_results.append(result_text)
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



def completion_model_workflow(model: ChatModel, info: Info, config) -> dict:
    eval_result = []
    for _ in range(config.pass_rate):
        # Prepare the utterance with context
        special_instr = [
            "You should think step by step before answering.",
            "If you need to perform calculations, do them carefully.",
            "Do not write python code as part of your answer. There is no code execution environment available."
        ]
        utterance_parts = [
            get_prompt(PROMPT_MODE) + "\n" + "\n".join(special_instr) + "\n" + get_response_format()
        ]
        if config.ingest_mode == "markdown":
            markdown = get_data_preview_markdown(info.data_file)
            utterance_parts.append(f"The data is as follows:\n{markdown}\n")
        utterance_parts.append(info.query)
        utterance = "\n".join(utterance_parts)
        user_message = Message(role=Role.User, content=utterance)
        response = model.chat(
            [user_message], 
            ChatRequest(max_completion_tokens=4096, n=1, temperature=0.0)
        )
        result_message = response.text
        try:
            agent_output = JsonResponseParser._parse_raw_response(result_message)
            agent_output = fix_json_serialization(agent_output)
        except:
            agent_output = {"answer": None, "dtype": None}
        eval = evaluate(
            gt_answer=info.answer,
            gt_dtype=info.dtype,
            pred_answer=agent_output['answer'],
            pred_dtype=agent_output['dtype']
        )
        eval_result.append({
            'agent_response': agent_output,
            'raw_response': result_message,
            'eval': eval,
            'tools': 0
        })
    info_dict = info.model_dump(mode="json")
    info_dict['eval'] = eval_result
    return info_dict

def without_tooling_workflow(model: ChatModel, info: Info, config) -> dict:
    """Run the workflow without tools using direct chat model.
    Uses model.chat() instead of agent for simpler interaction.
    """
    eval_result = []
    
    for _ in range(config.pass_rate):
        # Prepare the utterance with context
        utterance_parts = []
        
        # Add system prompt and response format
        system_message = Message(
            role=Role.System, 
            content=get_prompt(PROMPT_MODE) + "\n" + get_response_format()
        )
        
        # Add data preview if markdown mode
        if config.ingest_mode == "markdown":
            markdown = get_data_preview_markdown(info.data_file)
            utterance_parts.append(f"The data is as follows:\n{markdown}\n")
        
        # Add the actual query
        utterance_parts.append(info.query)
        
        utterance = "\n".join(utterance_parts)
        
        # Handle image files if in screenshot mode
        if config.ingest_mode == "screenshot" and info.image_file and info.image_file.exists():
            # Create message with image using Message object
            user_message = Message(role=Role.User, content=utterance, image=info.image_file)
        else:
            # Regular text-only message
            user_message = Message(role=Role.User, content=utterance)
        
        # Create messages list
        messages = [system_message, user_message]
        
        # Call model.chat() directly
        response = model.chat(
            messages, 
            # ChatRequest(max_completion_tokens=2048, n=1, temperature=0.0)
        )
        
        # Extract the response text
        result_message = response.text
        
        # Parse and evaluate
        try:
            agent_output = JsonResponseParser._parse_raw_response(result_message)
            agent_output = fix_json_serialization(agent_output)
        except:
            agent_output = {"answer": None, "dtype": None}
        eval = evaluate(
            gt_answer=info.answer,
            gt_dtype=info.dtype,
            pred_answer=agent_output['answer'],
            pred_dtype=agent_output['dtype']
        )
        eval_result.append({
            'agent_response': agent_output,
            'raw_response': result_message,
            'eval': eval,
            'tools': 0
        })
    
    info_dict = info.model_dump(mode="json")
    info_dict['eval'] = eval_result
    return info_dict


def tooling_workflow(model: ChatModel, info: Info, config) -> dict:
    """Run the agent workflow with the given info.
    Creates a fresh environment and agent for each run to ensure isolation.
    """
    eval_result = []
    
    for _ in range(config.pass_rate):
        # Create fresh environment for this run (only if sandbox is enabled)
        environment = CodeEnvironment()

        
        # Create fresh agent for this run
        # Only add tools if sandbox is enabled (not "no_sandbox" mode)
        agent = Agent(
            model=model,
            system=(get_prompt(PROMPT_MODE) + "\n" + get_response_format()),
            tools=[CodeSandboxTool()]
        )
        
        # Upload data file for this specific run (only if sandbox is enabled)
        if info.data_file and info.data_file.exists():
            environment.upload_file(info.data_file)
        
        # Prepare the utterance with context
        utterance_parts = []
        
        # Add data preview if markdown mode
        if config.ingest_mode == "markdown":
            markdown = get_data_preview_markdown(info.data_file)
            utterance_parts.append(f"The data is as follows:\n{markdown}\n")
        
        # Add file context (only if sandbox is enabled)
        if info.data_file:
            utterance_parts.append(f"The data file '{info.data_file.name}' has been uploaded to the code sandbox.\n")
        
        # Add the actual query
        utterance_parts.append(f"QUESTION: {info.query}")
        
        utterance = "\n".join(utterance_parts)
        
        # Handle image files if in screenshot mode
        if config.ingest_mode == "screenshot":
            # Create message with image using Message object
            message = Message(role=Role.User, content="The image of the data has been provided.", image=ROOT_DIR / info.image_file)
            environment.conversation.append(message)
        result = agent.run(utterance=utterance, environment=environment)
        
        try:
            agent_output = JsonResponseParser._parse_raw_response(result.message)
            agent_output = fix_json_serialization(agent_output)
        except:
            agent_output = {"answer": None, "dtype": None}
        eval = evaluate(
            gt_answer=info.answer,
            gt_dtype=info.dtype,
            pred_answer=agent_output['answer'],
            pred_dtype=agent_output['dtype']
        )
        eval_result.append({
            'agent_response': agent_output,
            'raw_response': result.message,
            'eval': eval,
            'tools': len(result.tools)
        })
    
    info_dict = info.model_dump(mode="json")
    info_dict['eval'] = eval_result
    return info_dict

def load_data(config) -> List['Info']:
    print(f"Using benchmark file: {config.input_file}")
    benchmark_info = Info.get_info(config.input_file)
    print(f"Got {len(benchmark_info)} benchmark infos")
    return benchmark_info


def get_config(args):
    # Define parser
    parser = argparse.ArgumentParser(description='Run evaluation on model')
    parser.add_argument('--input-file', default= ARTIFACT_DIR_SOURCE / f"{DATA_MODE}.json")
    parser.add_argument('--output-file', default=ROOT_DIR / "research" / "results" / datetime.datetime.now().strftime('%d%m%y') / f"{ARTIFACT_DIR_SOURCE.name}_{DATA_MODE}_{PROMPT_MODE}_{INGEST_MODE}_{MODEL}.json")
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




class CodeSandboxTool(Tool[CodeEnvironment]):
    """Tool that executes Python code in a sandboxed environment."""

    def definition(self) -> Function:
        return Function(
            name="execute_code",
            description="""
        Python Code Execution Sandbox. The Sandbox just executes a given python code and returns the stdout and stderr 
        of the executed code. Note that the sandbox just returns the code output and nothing else. 
        
        **IMPORTANT** Your response should always contain a single and complete python program. Do not write code snippets.
        
        It also contains the following uploaded files that can be used in the code:
        """,
            parameters={
                "code": StringProperty(
                    description="The Python code to execute. Should be a complete executable Python program.",
                ),
            },
        )

    def execute(
        self, arguments: dict[str, Any], environment: CodeEnvironment, identifier: str = None
    ) -> ToolResult:
        code = arguments.get("code", "")
        if not code:
            return ToolResult(values={"error": "No code provided to execute."})
        
        try:
            response = environment.code_tool.run_code(code)
            return ToolResult(
                values={
                    "exit_code": response.exit_code,
                    "output": response.log,
                    "output_files": [str(f) for f in (response.output_files or [])]
                }
            )
        except Exception as e:
            return ToolResult(values={"error": f"Code execution failed: {str(e)}"})

    def compile(self, result: ToolResult) -> str:
        if "error" in result.values:
            return f"Error: {result.values['error']}"
        
        exit_code = result.values.get("exit_code", 0)
        output = result.values.get("output", "")
        output_files = result.values.get("output_files", [])
        
        compiled_result = f"Code execution completed with exit code {exit_code}.\n"
        if output:
            compiled_result += f"Output:\n{output}\n"
        if output_files:
            compiled_result += f"Generated files: {', '.join(output_files)}"
        
        return compiled_result.strip()

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
            result = without_tooling_workflow(model, info, config)
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
