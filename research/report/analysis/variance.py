import sys
import argparse
sys.path.append(".")
from src.utils.utils import read_json

def get_variance(data):
    acc_scores = []
    for iter in range(len(data[0]["eval"])):
        total = 0
        success = 0
        for d in data:
            evals = d.get("eval", [])
            success += evals[iter].get("eval", False)
            total += 1
        acc_scores.append(success*100 / total if total > 0 else 0)
    mean = sum(acc_scores) / len(acc_scores) if acc_scores else 0
    variance = sum((x - mean) ** 2 for x in acc_scores) / len(acc_scores) if acc_scores else 0
    std_dev = variance ** 0.5
    return variance, std_dev


def main():
    parser = argparse.ArgumentParser(description="Analyze standard deviation from JSON files.")
    parser.add_argument("paths", type=str, nargs="+", help="Paths to the JSON files to read.")
    args = parser.parse_args()

    for path in args.paths:
        data = read_json(path)
        _, std_dev = get_variance(data)
        print(f"{std_dev:.2f}")

if __name__ == "__main__":
    main()
