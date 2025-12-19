import sys
sys.path.append(".")
from src.utils.utils import ROOT_DIR, read_json
from typing import List
import pandas as pd

def get_agg_score(dataset: List[dict]):
    total = 0
    success = 0
    for data in dataset:
        scores = data.get('eval', [])
        for score in scores:
            sc = score.get('eval', None)
            if sc is True:
                success += 1
                break
        total += 1
    agg_score = success / total if total > 0 else 0
    print(f"Aggregate Score: {agg_score*100:.2f}% ({success}/{total})")

def get_result_matrix(original_results: List[dict], diversified_results: List[dict], disturbed_results: List[dict]):
    # Determine the maximum number of eval results (for creating sheets)
    max_eval_length = 0
    for item in original_results + diversified_results + disturbed_results:
        eval_list = item.get('eval', [])
        if isinstance(eval_list, list):
            max_eval_length = max(max_eval_length, len(eval_list))
    
    # Create a dictionary for each eval index
    data_dicts = [{} for _ in range(max_eval_length)]
    summ_data_dicts = {}
    
    # Process original results
    for item in original_results:
        index = item['index']
        eval_list = item.get('eval', [])
        summ_data_dicts[index] = {
            'query_index': index,
            'query': item['query'],
            'data_file': item['data_file'],
            'original': False
        }
        
        # Add base info to all eval indices
        for eval_idx in range(len(eval_list)):
            if eval_idx < max_eval_length:
                if index not in data_dicts[eval_idx]:
                    data_dicts[eval_idx][index] = {
                        'query_index': index,
                        'query': item['query'],
                        'data_file': item['data_file'],
                    }
                # Extract the 'eval' value from the dict
                data_dicts[eval_idx][index]['original'] = eval_list[eval_idx].get('eval', None)
                summ_data_dicts[index]['original'] = eval_list[eval_idx].get('eval', None) or summ_data_dicts[index]['original']
    
    # Process diversified results
    for item in diversified_results:
        index = item['index']
        eval_list = item.get('eval', [])
        
        # Extract the base index and diversification type
        base_index = index[:index.find("__")]
        div_type = index[index.find("__")+2:]

        summ_data_dicts[base_index][div_type] = False
        
        for eval_idx in range(len(eval_list)):
            if eval_idx < max_eval_length and base_index in data_dicts[eval_idx]:
                # Extract the 'eval' value from the dict
                data_dicts[eval_idx][base_index][div_type] = eval_list[eval_idx].get('eval', None)
                summ_data_dicts[base_index][div_type] = eval_list[eval_idx].get('eval', None) or summ_data_dicts[base_index][div_type]
    
    # Process disturbed results
    for item in disturbed_results:
        index = item['index']
        eval_list = item.get('eval', [])
        
        # Extract the base index and diversification type
        base_index = index[:index.find("__")]
        div_type = index[index.find("__")+2:]
        summ_data_dicts[base_index][div_type] = False
        
        for eval_idx in range(len(eval_list)):
            if eval_idx < max_eval_length and base_index in data_dicts[eval_idx]:
                # Extract the 'eval' value from the dict
                data_dicts[eval_idx][base_index][div_type] = eval_list[eval_idx].get('eval', None)
                summ_data_dicts[base_index][div_type] = eval_list[eval_idx].get('eval', None) or summ_data_dicts[base_index][div_type]
    
    # Create Excel file with multiple sheets
    output_path = ROOT_DIR / "research" / "results" / "171225" / "summary_eval.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for eval_idx, data_dict in enumerate(data_dicts):
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data_dict, orient='index')
            
            # Reset index to make query_index a regular column
            df = df.reset_index(drop=True)
            
            # Reorder columns to have query_index, query, data_file, original first
            base_columns = ['query_index', 'query', 'data_file', 'original']
            other_columns = [col for col in df.columns if col not in base_columns]
            df = df[base_columns + other_columns]
            
            # Write to sheet (Sheet1, Sheet2, Sheet3, etc.)
            sheet_name = f'Sheet{eval_idx + 1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        # Write summary sheet
        summ_df = pd.DataFrame.from_dict(summ_data_dicts, orient='index')
        summ_df = summ_df.reset_index(drop=True)
        base_columns = ['query_index', 'query', 'data_file', 'original']
        other_columns = [col for col in summ_df.columns if col not in base_columns]
        summ_df = summ_df[base_columns + other_columns]
        summ_df.to_excel(writer, sheet_name='Summary', index=False)
    
    
    


if __name__ == "__main__":
    original_results = read_json(ROOT_DIR / "research" / "results" / "191225" / "original.json")
    try:
        diversified_results = read_json(ROOT_DIR / "research" / "results" / "191225" / "diversified.json")
    except AssertionError:
        diversified_results = []
    try:
        disturbed_results = read_json(ROOT_DIR / "research" / "results" / "191225" / "disturbed.json")
    except AssertionError:
        disturbed_results = []

    get_agg_score(original_results)
    get_agg_score(diversified_results)
    get_agg_score(disturbed_results)

    get_result_matrix(original_results, diversified_results, disturbed_results)


