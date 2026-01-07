import sys
sys.path.append(".")
import argparse
import json
import re
from pathlib import Path
from src.utils.utils import ROOT_DIR, read_json
from prose.llm import ChatModel, ChatRequest, Message, Role, SubstrateClient
from prose.llm.models import ModelSpecification, ModelSupports

PROMPT = """
You will be given an approach adopted by a model to help answering questions asked on a distorted table. The models was able to correctly answer the question on this distorted table. Looking at the approach, your job is to classify if the model detected the distortion and then formulated a solution or did it realise there was a distortion after it failed at some stage or felt the answer did not seem right. For the former case, label it as "before_solution" and for the latter case, label it as "after_failed_attempt".

Your output should be in the following format:
```json
{
  "distortion_detected": "before_solution" | "after_failed_attempt",
  "explanation": <brief explanation of your choice>
}
```
"""

def get_success(data):
    """Check if at least one evaluation passed."""
    scores = data.get('eval', [])
    for score in scores:
        sc = score.get('eval', None)
        if sc is True:
            return True
    return False

def extract_json_from_response(response_text):
    """Extract JSON from markdown code blocks or plain text."""
    # Try to find JSON in code blocks
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Try to parse the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None

distortion_ann = read_json(ROOT_DIR / "research" / "report" / "analysis" / "disturbance_annotation.json")

def analyze_distortion_detection(file_path):
    """Analyze how models detect distortions in successful cases."""
    # Read the result file
    data_list = read_json(file_path)
    
    # Initialize model
    model_name = ModelSpecification("dev-gpt-5-reasoning", ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_name, SubstrateClient(), suppress=True)
    
    # Filter successful cases
    successful_cases = [data for data in data_list if get_success(data)]
    
    print(f"Total cases: {len(data_list)}")
    print(f"Successful cases: {len(successful_cases)}")
    
    if len(successful_cases) == 0:
        print("No successful cases found.")
        return
    
    # Analyze each successful case
    before_solution_count = 0
    after_failed_attempt_count = 0
    failed_to_analyze = 0
    
    # Track by distortion_type
    distortion_type_stats = {}
    
    for idx, case in enumerate(successful_cases):
        print(f"\nAnalyzing case {idx + 1}/{len(successful_cases)}: {case.get('index', 'Unknown')}")
        
        # Get distortion_type
        distortion_type = case.get('distortion_type', 'unknown')
        
        if distortion_type == 'unknown':
            try:
                distortion_type = [d for d in distortion_ann if d['index'] == case['index']][0]['disturbance_annotation']
            except:
                distortion_type = 'unknown'

        
        # Get the raw response from the first successful evaluation
        raw_response = None
        for eval_item in case.get('eval', []):
            if eval_item.get('eval') is True:
                raw_response = eval_item.get('raw_response', '')
                break
        
        if not raw_response:
            print(f"  No raw response found for case {idx + 1}")
            failed_to_analyze += 1
            continue
        
        # Prepare prompt for LLM
        prompt_content = f"{PROMPT}\n\nQUESTION: {case['query']}\n\nApproach:\n{raw_response}"
        message = [Message(role=Role.User, content=prompt_content)]
        
        # Get LLM response
        try:
            response = model.chat(
                message, 
            )
            
            # Parse response
            result = extract_json_from_response(response.text)
            
            if result and 'distortion_detected' in result:
                detection_type = result['distortion_detected']
                explanation = result.get('explanation', '')
                
                print(f"  Distortion Type: {distortion_type}")
                print(f"  Detection: {detection_type}")
                print(f"  Explanation: {explanation}")
                
                # Initialize distortion_type stats if needed
                if distortion_type not in distortion_type_stats:
                    distortion_type_stats[distortion_type] = {
                        'before_solution': 0,
                        'after_failed_attempt': 0,
                        'failed': 0
                    }
                
                if detection_type == "before_solution":
                    before_solution_count += 1
                    distortion_type_stats[distortion_type]['before_solution'] += 1
                elif detection_type == "after_failed_attempt":
                    after_failed_attempt_count += 1
                    distortion_type_stats[distortion_type]['after_failed_attempt'] += 1
                else:
                    print(f"  Unexpected detection type: {detection_type}")
                    failed_to_analyze += 1
                    distortion_type_stats[distortion_type]['failed'] += 1
            else:
                print(f"  Failed to parse LLM response")
                failed_to_analyze += 1
                if distortion_type not in distortion_type_stats:
                    distortion_type_stats[distortion_type] = {
                        'before_solution': 0,
                        'after_failed_attempt': 0,
                        'failed': 0
                    }
                distortion_type_stats[distortion_type]['failed'] += 1
                
        except Exception as e:
            print(f"  Error analyzing case: {e}")
            failed_to_analyze += 1
            if distortion_type not in distortion_type_stats:
                distortion_type_stats[distortion_type] = {
                    'before_solution': 0,
                    'after_failed_attempt': 0,
                    'failed': 0
                }
            distortion_type_stats[distortion_type]['failed'] += 1
    
    # Calculate and display results
    total_analyzed = before_solution_count + after_failed_attempt_count
    
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Total successful cases: {len(successful_cases)}")
    print(f"Successfully analyzed: {total_analyzed}")
    print(f"Failed to analyze: {failed_to_analyze}")
    print(f"\nDistortion detected before solution: {before_solution_count}")
    print(f"Distortion detected after failed attempt: {after_failed_attempt_count}")
    
    if total_analyzed > 0:
        fraction = before_solution_count / total_analyzed
        percentage = fraction * 100
        print(f"\nFraction of 'before_solution' over total analyzed: {fraction:.4f} ({before_solution_count}/{total_analyzed})")
        print(f"Percentage of 'before_solution': {percentage:.2f}%")
    else:
        print("\nNo cases were successfully analyzed.")
    
    # Display results by distortion_type
    print("\n" + "="*60)
    print("RESULTS BY DISTORTION TYPE")
    print("="*60)
    
    for dtype in sorted(distortion_type_stats.keys()):
        stats = distortion_type_stats[dtype]
        total = stats['before_solution'] + stats['after_failed_attempt']
        
        print(f"\n{dtype.upper()}:")
        print(f"  Total analyzed: {total}")
        print(f"  Before solution: {stats['before_solution']}")
        print(f"  After failed attempt: {stats['after_failed_attempt']}")
        print(f"  Failed to analyze: {stats['failed']}")
        
        if total > 0:
            percentage = (stats['before_solution'] / total) * 100
            print(f"  Percentage 'before_solution': {percentage:.2f}% ({stats['before_solution']}/{total})")

def main():
    parser = argparse.ArgumentParser(description='Analyze distortion detection in successful cases')
    parser.add_argument('--file', type=str, required=True, help='Path to the result JSON file')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    analyze_distortion_detection(file_path)

if __name__ == "__main__":
    main()
