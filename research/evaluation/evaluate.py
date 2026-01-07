import pandas as pd
from typing import Any

ALLOWED_DTYPES = ["int", "float", "str", "list", "pd.Series", "set", "dict", "bool", "tuple"]

def _normalize_dtype(val: Any) -> str:
    """Normalize numpy/pandas types to base Python types."""
    dtype_name = type(val).__name__
    
    # Map numpy/pandas integer types
    if dtype_name in ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']:
        return 'int'
    
    # Map numpy/pandas float types
    if dtype_name in ['float64', 'float32', 'float16']:
        return 'float'
    
    # Map numpy/pandas bool
    if dtype_name in ['bool_', 'bool']:
        return 'bool'
    
    # Map numpy/pandas string
    if dtype_name in ['str_']:
        return 'str'
    
    return dtype_name

def evaluate(gt_answer: Any, gt_dtype: str, pred_answer: Any, pred_dtype: str) -> bool:
    if pred_dtype not in ALLOWED_DTYPES:
        return False
    
    try:
        if pred_dtype == "str":
            return str(gt_answer).lower().strip() == str(pred_answer).lower().strip()      
        
        pred_answer = eval(pred_answer) if isinstance(pred_answer, str) else pred_answer
        gt_answer = eval(gt_answer) if isinstance(gt_answer, str) else gt_answer



        if gt_dtype == "pd.Series":
            gt_series = pd.Series(gt_answer) if isinstance(gt_answer, dict) else gt_answer
            pred_series = pd.Series(pred_answer) if isinstance(pred_answer, dict) else pred_answer
            
            # Check length and index match
            if len(gt_series) != len(pred_series):
                return False
            # Normalize string indices to lowercase for comparison
            gt_index = gt_series.index.map(lambda x: x.lower() if isinstance(x, str) else x)
            pred_index = pred_series.index.map(lambda x: x.lower() if isinstance(x, str) else x)
            if not gt_index.equals(pred_index):
                return False
            
            # Recursively compare each value in the series
            for idx in gt_series.index:
                gt_val = gt_series[idx]
                pred_idx = [id for id in pred_series.index if (id.lower() if isinstance(id, str) else id) == (idx.lower() if isinstance(idx, str) else idx)][0]
                pred_val = pred_series[pred_idx]            
                if not evaluate(gt_val, _normalize_dtype(gt_val), pred_val, _normalize_dtype(pred_val)):
                    return False
            
            return True
        
        if gt_dtype == "int":
            return int(gt_answer) == int(pred_answer)
            
        if gt_dtype == "float":
            return abs(float(gt_answer) - float(pred_answer)) < 1e-2
        
        if gt_dtype == "bool":
            return bool(gt_answer) == bool(pred_answer)
            
        if gt_dtype == "list" or gt_dtype == "tuple":
            if len(gt_answer) != len(pred_answer):
                return False
            
            # Create a mutable copy of pred_answer to track matches
            remaining_pred = list(pred_answer)
            
            for gt_item in gt_answer:
                match_found = False
                for i, pred_item in enumerate(remaining_pred):
                    if evaluate(gt_item, _normalize_dtype(gt_item), pred_item, _normalize_dtype(pred_item)):
                        remaining_pred.pop(i)
                        match_found = True
                        break
                if not match_found:
                    return False
            
            return True
            
        if gt_dtype == "set":
            return set(gt_answer) == set(pred_answer)
        
        if gt_dtype == "dict":
            if len(gt_answer) != len(pred_answer):
                return False
            gt_answer = {str(k).lower() if isinstance(k, str) else k: v for k, v in gt_answer.items()}
            pred_answer = {str(k).lower() if isinstance(k, str) else k: v for k, v in pred_answer.items()}
            for key in gt_answer:
                if key not in pred_answer:
                    return False
                if not evaluate(gt_answer[key], _normalize_dtype(gt_answer[key]), pred_answer[key], _normalize_dtype(pred_answer[key])):
                    return False
            return True
    except:
        return False
        
    return False
