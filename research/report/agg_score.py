import sys
sys.path.append(".")
from src.utils.utils import ROOT_DIR, read_json
from typing import List
import pandas as pd

def get_agg_score(dataset: List[dict]):
    total = 0
    success = 0
    for data in dataset:
        score = data.get('eval', None)
        if score is None:
            continue
        total += 1
        if score is True:
            success += 1
    agg_score = success / total if total > 0 else 0
    print(f"Aggregate Score: {agg_score*100:.2f}% ({success}/{total})")

def get_result_matrix(original_results: List[dict], diversified_results: List[dict], disturbed_results: List[dict]):
    # Create a dictionary to store data by query index
    data_dict = {}
    
    # Process original results
    for item in original_results:
        index = item['index']
        data_dict[index] = {
            'query_index': index,
            'query': item['query'],
            'data_file': item['data_file'],
            'original': item['eval']
        }
    
    # Process diversified results
    for item in diversified_results:
        index = item['index']
        # Extract the base index and diversification type
        base_index = index[:index.find("__")]
        div_type = index[index.find("__")+2:]
        
        if base_index in data_dict:
            data_dict[base_index][div_type] = item['eval']
    
    # Process disturbed results
    for item in disturbed_results:
        index = item['index']
        # Extract the base index and diversification type
        base_index = index[:index.find("__")]
        div_type = index[index.find("__")+2:]
        
        if base_index in data_dict:
            data_dict[base_index][div_type] = item['eval']
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    
    # Reset index to make query_index a regular column
    df = df.reset_index(drop=True)
    
    # Reorder columns to have query_index, query, data_file, original first
    base_columns = ['query_index', 'query', 'data_file', 'original']
    other_columns = [col for col in df.columns if col not in base_columns]
    df = df[base_columns + other_columns]

    df.to_excel(ROOT_DIR / "research" / "results" / "071225" / "summary_eval.xlsx", index=False, engine='openpyxl')
    
    


if __name__ == "__main__":
    original_results = read_json(ROOT_DIR / "research" / "results" / "071225" / "original_evaluated.json")
    diversified_results = read_json(ROOT_DIR / "research" / "results" / "071225" / "diversified_evaluated.json")
    disturbed_results = read_json(ROOT_DIR / "research" / "results" / "071225" / "disturbed_evaluated.json")

    get_agg_score(original_results)
    get_agg_score(diversified_results)
    get_agg_score(disturbed_results)

    get_result_matrix(original_results, diversified_results, disturbed_results)


