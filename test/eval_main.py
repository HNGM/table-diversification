import sys
sys.path.append(".")
from research.evaluation.prose_llm_main import CodeEnvironment
from src.utils.utils import read_json, ROOT_DIR
from src.utils.llm_config import LLMConfig, load_llm_configs
from research.agents.sandbox_agent import FunctionCallAdaAgent
from research.agents.no_function_call_agent import NoFunctionCallAdaAgent
from research.evaluation.utils import get_prompt
from research.agents.output_format import get_response_format, get_python_response_format
from src.interfaces import UserMessage, Message
from research.evaluation.info import Info
from src.utils.data_preview import get_data_preview_markdown
from research.agents.instr_chat_agent import InstrChatAgent
from prose.llm import ChatModel, ChatRequest, Message, Role, SubstrateClient, SubstrateModels, Function
from prose.llm.models import ModelSpecification, ModelSupports
from research.agents.utils.model_response import PythonResponseParser, JsonResponseParser
from research.agents.utils.code_tool import CodeTool
import re
import json
from research.evaluation.utils import fix_json_serialization


PROMPT = """
You are Dazza, an expert data analyst specializing in solving data analytics questions based on a tabular data. Your role is to analyze the data and provide accurate answers to user queries.

**Your Environment:**
- You will be provided with the relevant data context required to answer the query.

**Your Responsibilities:**
1. Carefully analyze the user's query to understand what information they need
2. Study the table structure and content to better understand how to answer queries accurately
3. Provide a direct answer based on your analysis. Do not suggest approaches or offer partial solutions

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
"""


def test_eval_workflow():
    llm_config = load_llm_configs(ROOT_DIR / "config" / "default_llm_config.json", "dev-deepseek-r1-distill-qwen-32b")[0]
    query = "For each doctor, what proportion of their visits required a follow-up?"
    data_file = ROOT_DIR / r"dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.xlsx"
    image_file = ROOT_DIR / r"dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.png"
    msgs = []
    agent = NoFunctionCallAdaAgent(llm_config, prompt=get_prompt("default_no_sandbox") + "\n" + get_response_format())    
    # msgs.append(agent.upload_image_files(image_files=[image_file], metadata="The image of the table in the data file provided. Use this to make computations."))
    msgs.append(UserMessage(content=get_data_preview_markdown(data_file)))
    msgs.append(UserMessage(content=query))
    agent_output = agent.run(msgs)
    print(agent_output.ParsedResponse)
    print("\n--- RAW RESPONSE ---\n")
    print(agent_output.RawResponse)

def test_simple_qwen():
    llm_config = load_llm_configs(ROOT_DIR / "config" / "default_llm_config.json", "tablegpt2-7b-local")[0]
    query = "What is the capital of France?"
    msgs = []
    agent = FunctionCallAdaAgent(llm_config=llm_config, prompt="You are a helpful assistant.")
    msgs.append(UserMessage(content=query))
    agent_output = agent.run(msgs)
    print(agent_output.ParsedResponse)
    print("\n--- RAW RESPONSE ---\n")
    print(agent_output.RawResponse)

def test_image_reading():
    llm_config = load_llm_configs(ROOT_DIR / "config" / "default_llm_config.json", "dev-qwen-25-vl7b")[0]
    query = "How many columns are there in the image of the table provided?"
    image_file = ROOT_DIR / r"dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.png"
    msgs = []
    agent = InstrChatAgent(llm_config, prompt="Do what the user says.")
    msgs.append(agent.upload_image_files(image_files=[image_file], metadata="The image of the table in the data file provided. Use this to make computations."))
    msgs.append(UserMessage(content=query))
    agent_output = agent.run(msgs)
    print(agent_output.ParsedResponse)

def test_image_reading_prose_llm():
    query = "How many columns are there in the image of the table provided?"
    image_file = ROOT_DIR / r"dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.png"
    user_message = [Message(role=Role.User, content=query, images=image_file)]
    model_name = ModelSpecification("dev-deepseek-r1-distill-qwen-32b", ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)
    response = model.chat(
        user_message, 
        ChatRequest(max_completion_tokens=2048, n=1, temperature=0.0)
    )
    print(response.text)

def test_prose_llm():
    query = "For each doctor, what proportion of their visits required a follow-up?"
    data_file = ROOT_DIR / r"dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.xlsx"
    markdown = get_data_preview_markdown(data_file)
    message = [
        Message(role=Role.User, content=f"The data preview is as follows:\n{markdown}\nquery: {query}")
    ]
    model_name = ModelSpecification("dev-deepseek-r1-distill-qwen-32b", ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)
    response = model.chat(
        message, 
        ChatRequest(max_completion_tokens=2048, n=1, temperature=0.0)
    )
    print(response.text)

def test_tool_calling_prose_llm():
    query = "For each doctor, what proportion of their visits required a follow-up?"
    data_file = ROOT_DIR / r"dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.xlsx"
    
    # Create code tool and upload data file
    code_tool = CodeTool()
    code_tool.upload_files([data_file])
    
    # Get data preview
    markdown = get_data_preview_markdown(data_file)
    
    # Initialize messages
    messages = [
        Message(role=Role.System, content=get_prompt("default_tool_calling") + "\n" + get_python_response_format()),
        Message(role=Role.User, content=f"The data file '{data_file.name}' is available in the sandbox.\n\nThe data is as follows:\n{markdown}\n\nQuery: {query}")
    ]
    
    # Create model
    model_name = ModelSpecification("dev-deepseek-r1-distill-qwen-32b", ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)
    
    # Iterative execution loop
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print('='*60)
        
        # Get model response
        response = model.chat(
            messages, 
            ChatRequest(max_completion_tokens=2048, n=1, temperature=0.0)
        )
        response_text = response.text
        print(f"\nModel response:\n{response_text}\n")
        
        # Extract ```python code blocks first (prioritized over json)
        python_pattern = r'```python\s*(.*?)\s*```'
        python_matches = re.findall(python_pattern, response_text, re.DOTALL)
        
        if python_matches:
            print(f"Found {len(python_matches)} Python code block(s)")
            
            # Execute all Python code blocks
            execution_results = []
            for idx, code_block in enumerate(python_matches, 1):
                print(f"\nExecuting code block {idx}...")

                code_response = code_tool.run_code(code_block)
                if code_response.exit_code == 0:
                    result_text = f"✓ Code block {idx} executed successfully:\n{code_response.log}" + get_response_format()
                    print(result_text) 
                    execution_results.append(result_text)
                else:
                    result_text = f"✗ Code block {idx} failed (exit code {code_response.exit_code}):\n{code_response.log}"
                    print(result_text)
                    execution_results.append(result_text)
                
            
            # Append assistant response and execution results to messages
            messages.append(Message(role=Role.Assistant, content=response_text))
            messages.append(Message(
                role=Role.User,
                content="Execution results:\n\n" + "\n\n".join(execution_results)
            ))
            continue
        
        # Check for ```json block (final answer) only if no python code found
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            print("\n✓ Found final answer in ```json block!")
            try:
                json_text = json_match.group(1)
                final_answer = json.loads(json_text)
                print(f"Final answer: {final_answer}")
                return final_answer
            except Exception as e:
                print(f"Failed to parse JSON: {e}")
                break
        
        # No python code and no json - prompt for final answer
        print("⚠ No Python code or JSON found, prompting for final answer...")
        messages.append(Message(role=Role.Assistant, content=response_text))
        messages.append(Message(
            role=Role.User,
            content="Please provide your final answer in the ```json format."
        ))
    
    print("\n⚠ Max iterations reached without finding final answer")
    return None

def test_tool_calling_completion_prose_llm():
    """Test iterative code execution with completion model."""
    # Setup test data
    query = "For each doctor, what proportion of their visits required a follow-up?"
    data_file = ROOT_DIR / r"dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Disturbed Diversifications/Clinic_Visits/Clinic_Visits_original_1.xlsx"
    
    # Create code tool and upload data file
    code_tool = CodeTool()
    code_tool.upload_files([data_file])
    
    # Initialize messages
    messages = [
        get_prompt("default_tool_calling"),
        get_python_response_format()
    ]
    
    # Add data preview
    markdown = get_data_preview_markdown(data_file)
    messages.append(f"The data is as follows:\n{markdown}\nIt has also been uploaded to the sandbox by the name '{data_file.name}'.\n")
    
    # Add query
    messages.append(query)
    
    # Create model
    model_name = ModelSpecification("dev-deepseek-r1-distill-qwen-32b", ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)
    
    # Iterative execution loop
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print('='*60)
        
        # Get model response
        response = model.chat(
            [Message(role=Role.User, content="\n".join(messages))], 
            ChatRequest(max_completion_tokens=4096, n=1, temperature=0.0)
        )
        result_message = response.text
        print(f"\nModel response:\n{result_message[:500]}...\n")
        
        # Extract ```python code blocks first
        python_pattern = r'```python\s*(.*?)\s*```'
        python_matches = re.findall(python_pattern, result_message, re.DOTALL)

        if python_matches:
            print(f"Found {len(python_matches)} Python code block(s)")
            execution_results = []
            for idx, code_block in enumerate(python_matches, start=1):
                print(f"\nExecuting code block {idx}...")
                try:
                    code_response = code_tool.run_code(code_block)
                    if code_response.exit_code == 0:
                        result_text = f"Code block {idx} executed successfully:\n{code_response.log}"
                        print(f"✓ {result_text}")
                    else:
                        result_text = f"Code block {idx} failed (exit code {code_response.exit_code}):\n{code_response.log}"
                        print(f"✗ {result_text}")
                    execution_results.append(result_text)
                except Exception as e:
                    result_text = f"Code block {idx} error: {str(e)}"
                    print(f"✗ {result_text}")
                    execution_results.append(result_text)
            
            messages.append(result_message)
            messages.append(f"Sandbox execution results:\n" + "\n".join(execution_results) + "\n\n" + get_response_format())
            continue
        
        # Check for ```json block (final answer) only if no python code found
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, result_message, re.DOTALL)
        
        if json_match:
            print("\n✓ Found final answer in ```json block!")
            try:
                agent_output = JsonResponseParser._parse_raw_response(result_message)
                agent_output = fix_json_serialization(agent_output)
                print(f"Parsed answer: {agent_output}")
                
                # Note: For testing without real ground truth, we'll just return the answer
                # In production, you would have info.answer and info.dtype
                return {
                    'agent_response': agent_output,
                    'raw_response': "\n".join(messages),
                    'iterations': iteration + 1
                }
            except Exception as e:
                print(f"Failed to parse JSON: {e}")
                break
        
        # No python code and no json
        print("⚠ No Python code or JSON found, prompting for final answer...")
        messages.append(result_message)
        messages.append("The previous response did not contain valid Python code or a final JSON answer. Please try again.")
    
    print("\n⚠ Max iterations reached without finding final answer")
    return None

def test_multiple_tool_calls_prose_llm():
    """Test if a model can make multiple tool calls in the same response using prose.llm Agent."""
    from research.evaluation.prose_llm_main import CodeEnvironment, CodeSandboxTool
    from prose.llm.agent import Agent
    
    # Create model
    model_name = ModelSpecification("dev-gpt-5-reasoning", ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)
    
    # Create environment with code tool
    environment = CodeEnvironment()
    
    # Create agent with code execution tool
    agent = Agent(
        model=model,
        system="You are a helpful assistant with access to a Python code execution sandbox.",
        tools=[CodeSandboxTool()]
    )
    
    # Query that explicitly asks for multiple tool calls
    query = """Execute the following Python code snippets separately using multiple tool calls:

1. First tool call: Print the numbers 1 through 5
2. Second tool call: Calculate and print 10 + 20
3. Third tool call: Print the current date and time

Please make THREE separate execute_code tool calls, one for each task."""
    
    # Run the agent
    print("Running agent with query that should trigger multiple tool calls...")
    result = agent.run(utterance=query, environment=environment)
    
    # Display results
    print(f"\n{'='*60}")
    print("Agent Response:")
    print('='*60)
    print(result.message)
    
    print(f"\n{'='*60}")
    print(f"Number of Tool Calls: {len(result.tools)}")
    print('='*60)
    
    for idx, tool_call in enumerate(result.tools, 1):
        print(f"\nTool Call {idx}:")
        print(f"  Tool: {tool_call.name if hasattr(tool_call, 'name') else 'N/A'}")
        print(f"  Result preview: {str(tool_call)[:200]}...")
    
    print(f"\n{'='*60}")
    print(f"Test Result: {'✓ PASSED' if len(result.tools) > 1 else '✗ FAILED'} - Made {len(result.tools)} tool call(s)")
    print('='*60)
    
    return result


def test_tool_calling():
    """Test tool calling capability with a weather query using custom tool definition format."""
    # Create model
    model_name = ModelSpecification("dev-deepseek-r1-distill-qwen-32b", ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)
    
    # Define messages with tool calling format
    messages = [
        Message(
            role=Role.System,
            content="""You are a helpful assistant that can use tools to get information for the user.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "get_weather", "description": "Get current weather information for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to use"}}, "required": ["location"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags."""
        ),
        Message(
            role=Role.User,
            content="What's the weather like in New York?"
        )
    ]
    
    # Make the request
    print("Testing tool calling with weather query...")
    print(f"\n{'='*60}")
    print("User Query: What's the weather like in New York?")
    print('='*60)
    
    response = model.chat(
        messages,
        ChatRequest(max_completion_tokens=2048, n=1, temperature=0.0)
    )
    
    # Display the response
    print(f"\nModel Response:")
    print('-'*60)
    print(response.text)
    print('-'*60)
    
    # Check if the response contains tool call
    if '<tool_call>' in response.text and '</tool_call>' in response.text:
        print("\n✓ Model successfully generated tool call!")
        
        # Extract tool call
        import re
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_calls = re.findall(tool_call_pattern, response.text, re.DOTALL)
        
        print(f"\nFound {len(tool_calls)} tool call(s):")
        for idx, call in enumerate(tool_calls, 1):
            print(f"\nTool Call {idx}:")
            try:
                call_json = json.loads(call.strip())
                print(json.dumps(call_json, indent=2))
            except:
                print(call)
    else:
        print("\n✗ No tool calls found in response")
    
    return response



if __name__ == "__main__":
    # test_multiple_tool_calls_prose_llm()
    test_tool_calling()
