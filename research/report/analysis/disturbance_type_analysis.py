import sys
sys.path.append(".")
from src.utils.utils import ROOT_DIR, read_json
from typing import List
from research.evaluation.stats import get_tool_call_count, get_tool_call_count_completion_prosellm
import argparse

DISTB_ANN = read_json(ROOT_DIR / "research" / "report" / "analysis" / "disturbance_annotation.json")

def get_disturbance_type_score(results: List[dict]):
    distb_score = {}
    for item in results:
        distb_type = [distb for distb in DISTB_ANN if distb['index'] == item['index']][0]['disturbance_annotation']
        if distb_type not in distb_score:
            distb_score[distb_type] = []
        distb_score[distb_type].append({
            "success": any([eval_item.get('eval', False) is True for eval_item in item.get('eval', [])]),
            "tool_call": sum([eval_item.get("tools", 0) for eval_item in item.get('eval', [])]) / len(item.get('eval', [])) if len(item.get('eval', [])) > 0 else 0
        })
        if not distb_score[distb_type][-1]["success"]:
            print(f"Failed Index: {item['index']} | Disturbance Type: {distb_type}")
    agg_distb_score = {}
    for distb_type, records in distb_score.items():
        total = len(records)
        success = sum([1 for record in records if record['success']])
        avg_tool_call = sum([record['tool_call'] for record in records]) / total if total > 0 else 0
        agg_distb_score[distb_type] = {
            "aggregate_score": success / total if total > 0 else 0,
            "total": total,
            "success": success,
            "avg_tool_call": avg_tool_call
        }
    print("Disturbance Type Analysis:")
    for distb_type, scores in agg_distb_score.items():
        print(f"- {distb_type.capitalize()}: {scores['aggregate_score']*100:.2f}% ({scores['success']}/{scores['total']}), Avg Tool Calls: {scores['avg_tool_call']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze disturbance types from results')
    parser.add_argument('--file', type=str, default=r"research\results\231225\disturbed_default_mistake_no_sandbox_markdown_dev-gpt-52-reasoning.json", help='Path to the results JSON file')
    args = parser.parse_args()
    
    disturbed_results = read_json(args.file)
    get_disturbance_type_score(disturbed_results)
    
