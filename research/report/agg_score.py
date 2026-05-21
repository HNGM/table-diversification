import sys
import os
sys.path.append(".")
from src.utils.utils import ROOT_DIR, read_json
from typing import List, Optional
from collections import defaultdict
import pandas as pd
import argparse


WIKITQ_ORIGINAL_PREFIX = "wikitq_dataset_"
WIKITQ_DISTURBED_DATASET = ROOT_DIR / "research" / "dataset" / "wikitq_dataset_filtered" / "disturbed.json"


def _build_query_to_disturbed(disturbed_dataset_path=WIKITQ_DISTURBED_DATASET):
    """Group disturbed dataset entries by their original query."""
    disturbed = read_json(disturbed_dataset_path)
    by_query = defaultdict(list)
    for d in disturbed:
        by_query[d.get('query')].append(d)
    return by_query


def _extract_eval(data):
    """Return (is_success, avg_tool_calls) for a single result entry."""
    scores = data.get('eval', [])
    tl = []
    is_success = False
    for score in scores:
        sc = score.get('eval', None)
        tl.append(score.get('tools', 0))
        if sc is True:
            is_success = True
    avg_tools = sum(tl) / len(tl) if tl else 0
    return is_success, avg_tools


def get_agg_score(dataset: List[dict], scale_map: Optional[dict] = None):
    total = 0
    success = 0
    tool = 0.0
    for data in dataset:
        is_success, avg_tools = _extract_eval(data)
        if scale_map is not None:
            weight = len(scale_map.get(data.get('query'), []))
        else:
            weight = 1
        if weight == 0:
            continue
        total += weight
        if is_success:
            success += weight
        tool += avg_tools * weight
    agg_score = success / total if total > 0 else 0
    print(f"Aggregate Score: {agg_score*100:.2f}% ({success}/{total})")
    print(f"Average Tool Calls: {tool/total:.2f}" if total > 0 else "Average Tool Calls: N/A")


def get_distortion_type_score(dataset: List[dict], scale_map: Optional[dict] = None):
    distortion_score = {}

    if scale_map is not None:
        # Scaled mode: bin each original result by the distortion_type of every
        # matching disturbed variant (counts success/total per variant).
        for data in dataset:
            is_success, avg_tools = _extract_eval(data)
            variants = scale_map.get(data.get('query'), [])
            for variant in variants:
                dist_type = variant.get('distortion_type', 'unknown')
                bucket = distortion_score.setdefault(
                    dist_type, {'total': 0, 'success': 0, 'tool_calls': 0.0}
                )
                bucket['total'] += 1
                if is_success:
                    bucket['success'] += 1
                bucket['tool_calls'] += avg_tools
    else:
        for data in dataset:
            distortion_type = data.get('distortion_type', 'unknown')
            bucket = distortion_score.setdefault(
                distortion_type, {'total': 0, 'success': 0, 'tool_calls': 0.0}
            )
            is_success, avg_tools = _extract_eval(data)
            bucket['total'] += 1
            if is_success:
                bucket['success'] += 1
            bucket['tool_calls'] += avg_tools

    print("Distortion Type Analysis:")
    for dist_type, scores in distortion_score.items():
        total = scores['total']
        success = scores['success']
        avg_tool_calls = scores['tool_calls'] / total if total > 0 else 0
        agg_score = success / total if total > 0 else 0
        print(f"- {dist_type.capitalize()}: {agg_score*100:.2f}% ({success}/{total}), Avg Tool Calls: {avg_tool_calls:.2f}")


def get_diversification_breakup(dataset: List[dict], scale_map: Optional[dict] = None):
    """Per-diversification-type accuracy, grouped by distortion_type.

    Shows ``success/total`` as a fraction first, then the percentage.
    """
    # buckets[distortion_type][diversification_type] = {'total','success','tool_calls'}
    buckets: dict = {}

    def _ensure(dist_t, div_t):
        return buckets.setdefault(dist_t, {}).setdefault(
            div_t, {'total': 0, 'success': 0, 'tool_calls': 0.0}
        )

    if scale_map is not None:
        for data in dataset:
            is_success, avg_tools = _extract_eval(data)
            for variant in scale_map.get(data.get('query'), []):
                dist_t = variant.get('distortion_type', 'unknown')
                div_t = variant.get('diversification_type', 'unknown')
                b = _ensure(dist_t, div_t)
                b['total'] += 1
                if is_success:
                    b['success'] += 1
                b['tool_calls'] += avg_tools
    else:
        for data in dataset:
            dist_t = data.get('distortion_type', 'unknown')
            div_t = data.get('diversification_type', 'unknown')
            is_success, avg_tools = _extract_eval(data)
            b = _ensure(dist_t, div_t)
            b['total'] += 1
            if is_success:
                b['success'] += 1
            b['tool_calls'] += avg_tools

    # Stable order: structural first, then semantic, then any others alphabetically.
    preferred = ['structural', 'semantic']
    ordered_dts = [d for d in preferred if d in buckets] + sorted(
        [d for d in buckets if d not in preferred]
    )

    print("\nDiversification Type Breakup (grouped by distortion_type):")
    for dist_t in ordered_dts:
        divs = buckets[dist_t]
        # Bucket totals across diversification types.
        dt_total = sum(v['total'] for v in divs.values())
        dt_success = sum(v['success'] for v in divs.values())
        dt_tools = sum(v['tool_calls'] for v in divs.values())
        dt_pct = (dt_success / dt_total * 100) if dt_total else 0.0
        dt_avg_tools = (dt_tools / dt_total) if dt_total else 0.0
        print(
            f"\n[{dist_t.capitalize()}] overall: {dt_success}/{dt_total} "
            f"= {dt_pct:.2f}%, Avg Tool Calls: {dt_avg_tools:.2f}"
        )
        # Sort diversification types by total desc, then name.
        for div_t, sc in sorted(divs.items(), key=lambda kv: (-kv[1]['total'], kv[0])):
            total = sc['total']
            success = sc['success']
            pct = (success / total * 100) if total else 0.0
            avg_tools = (sc['tool_calls'] / total) if total else 0.0
            print(
                f"  - {div_t}: {success}/{total} = {pct:.2f}%, "
                f"Avg Tool Calls: {avg_tools:.2f}"
            )

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
    parser = argparse.ArgumentParser(description='Get Aggregated score from results')
    parser.add_argument('--file', type=str, default=r"research\results\231225\disturbed_default_mistake_no_sandbox_markdown_dev-gpt-52-reasoning.json", help='Path to the results JSON file')
    parser.add_argument('--breakup', action='store_true', help='Also show accuracy broken down by diversification_type (grouped by distortion_type).')
    args = parser.parse_args()

    disturbed_results = read_json(args.file)

    base_name = os.path.basename(args.file)
    scale_map = None
    if base_name.startswith(WIKITQ_ORIGINAL_PREFIX):
        print(f"Detected wikitq original results file '{base_name}'. "
              f"Scaling scores by disturbed.json variant counts.")
        scale_map = _build_query_to_disturbed()

    get_agg_score(disturbed_results, scale_map=scale_map)
    get_distortion_type_score(disturbed_results, scale_map=scale_map)
    if args.breakup:
        get_diversification_breakup(disturbed_results, scale_map=scale_map)

    # get_result_matrix(original_results, diversified_results, disturbed_results)


