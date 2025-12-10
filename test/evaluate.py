import sys
sys.path.append(".")
from research.evaluation.evaluate import evaluate
from src.utils.utils import read_json, ROOT_DIR, write_json
import traceback

def test():
    data = read_json(ROOT_DIR / "research" / "results" / "071225" / "original.json")
    for item in data:
        try:
            result = evaluate(
                gt_answer=item['answer'],
                gt_dtype=item['dtype'],
                pred_answer=item['agent_response']['answer'],
                pred_dtype=item['agent_response']['dtype']
            )
        except Exception as e:
            print(traceback.format_exc())
            continue
        item['eval'] = result
    write_json(data, ROOT_DIR / "research" / "results" / "071225" / "original_evaluated.json")

if __name__ == "__main__":
    pred_answer = {
        "West": 11625.25,
        "East": 2617.5,
        "South": 1209.0,
        "North": 600.0
    }
    result = evaluate(
        "{'West': 11625.25, 'East': 2617.50, 'South': 1209.00, 'North': 600.00}",
        "pd.Series",
        pred_answer,
        "pd.Series"
    )
    print(result)