import sys
sys.path.append(".")
from src.utils.utils import ROOT_DIR, read_json
from typing import List

def get_agg_score(dataset: List[dict]):
    total = 0
    success = 0
    for data in dataset:
        scores = data.get('evaluation', {}).get('verdict', None)
        if scores is None:
            continue
        total += 1
        if scores is True:
            success += 1
    agg_score = success / total if total > 0 else 0
    print(f"Aggregate Score: {agg_score*100:.2f}% ({success}/{total})")

if __name__ == "__main__":
    dataset_path = ROOT_DIR / "research" / "results" / "251125" / "202849.json"
    dataset = read_json(dataset_path)
    get_agg_score(dataset)

