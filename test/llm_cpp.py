# !pip install llama-cpp-python

from llama_cpp import Llama
import json
from pathlib import Path
import sys
sys.path.append(".")

def test():
    llm = Llama.from_pretrained(
        repo_id="tensorblock/RUCKBReasoning_TableLLM-7b-GGUF",
        filename="TableLLM-7b-Q2_K.gguf",
        n_ctx=4096,  # Increase context window size (default is 512)
        n_batch=512,  # Number of tokens to process in parallel
        verbose=False,  # Disable verbose logging
    )

    response = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )

    print(response['choices'][0]['message']['content'])


def test_tool_calling():
    """Example of tool calling with llama-cpp-python."""
    llm = Llama.from_pretrained(
        repo_id="QuantFactory/TableGPT2-7B-GGUF",
        filename="TableGPT2-7B.Q2_K.gguf",
        n_ctx=4096,  # Increase context window to handle longer prompts
        n_batch=512,  # Batch size for prompt processing
        verbose=False,  # Suppress meta information and loading messages
    )
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature to use"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate, e.g. '2 + 2'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    # Create chat completion with tools
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use the tools when necessary to answer questions."
        },
        {
            "role": "user",
            "content": "What's the weather like in New York and what is 15 * 23?"
        }
    ]
    
    response = llm.create_chat_completion(
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Let the model decide when to use tools
    )
    
    print("Response:")
    print(json.dumps(response, indent=2))
    
    # Check if the model made tool calls
    if 'tool_calls' in response['choices'][0]['message']:
        tool_calls = response['choices'][0]['message']['tool_calls']
        print(f"\n{len(tool_calls)} tool call(s) detected:")
        
        for idx, tool_call in enumerate(tool_calls, 1):
            print(f"\nTool Call {idx}:")
            print(f"  Function: {tool_call['function']['name']}")
            print(f"  Arguments: {tool_call['function']['arguments']}")
            
        # Simulate tool execution and continue conversation
        messages.append(response['choices'][0]['message'])
        
        for tool_call in tool_calls:
            # Simulate tool results
            if tool_call['function']['name'] == 'get_weather':
                tool_result = {"temperature": 72, "condition": "sunny", "unit": "fahrenheit"}
            elif tool_call['function']['name'] == 'calculate':
                args = json.loads(tool_call['function']['arguments'])
                tool_result = {"result": eval(args['expression'])}
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "content": json.dumps(tool_result)
            })
        
        # Get final response
        final_response = llm.create_chat_completion(messages=messages)
        print("\nFinal Response:")
        print(final_response['choices'][0]['message']['content'])
    else:
        print("\nNo tool calls made")
        print("Content:", response['choices'][0]['message']['content'])

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

def test_tablellm_tooling():
    from src.utils.data_preview import get_data_preview_markdown
    from src.utils.utils import ROOT_DIR
    sample_data = {
        "query": "For each doctor, what proportion of their visits required a follow-up?",
        "answer": "{\"Dr. Iyer\": 0.5, \"Dr. Rao\": 0.5, \"Dr. Singh\": 0.3333333333333333}",
        "dtype": "dict",
        "data_file": "dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.xlsx",
    }
    llm = Llama.from_pretrained(
        repo_id="tensorblock/RUCKBReasoning_TableLLM-7b-GGUF",
        filename="TableLLM-7b-Q2_K.gguf",
        n_ctx=4096,  # Increase context window size (default is 512)
        n_batch=512,  # Number of tokens to process in parallel
        verbose=False,  # Disable verbose logging
    )

    markdown = get_data_preview_markdown(ROOT_DIR / sample_data['data_file'])

    response = llm.create_chat_completion(
        messages = [
            {
                "role": "user", 
                "content": TOOLING_PROMPT.replace("{csv_data}", markdown).replace("{question}", sample_data['query']).replace("{data_file}", Path(sample_data['data_file']).name).replace("{mistake_instr}", ""),
            }
        ]
    )
    print(response['choices'][0]['message']['content'])

def test_ans_extractor():
    raw_ans = "Average hours worked remotely: 7.423076923076923, Average hours worked onsite: 8.071428571428571"
    from research.evaluation.tablellm import AnsExtractorPrompt, ans_extract_model, get_response_format, proseMessage, Role
    response = ans_extract_model.chat([
        proseMessage(role=Role.User, content=AnsExtractorPrompt.replace("{user_question}", "What is the average number of hours worked for employees working remotely compared to those onsite?").replace("{code_output}", raw_ans) + get_response_format())
    ])
    print(response.text)


if __name__ == "__main__":
    test_ans_extractor()