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
        "answer": {
            "europe": "volkswagen 1131 deluxe sedan",
            "japan": "toyota corona mark ii",
            "usa": "ford maverick"
        },
        "dtype": "dict"
    }
    result = evaluate(
        "{'usa': 'ford maverick', 'japan': 'toyota corona mark ii', 'europe': 'volkswagen 1131 deluxe sedan'}",
        "dict",
        pred_answer['answer'],
        pred_answer['dtype']
    )
    print(result)