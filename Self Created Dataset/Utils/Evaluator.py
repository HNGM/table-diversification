import pandas as pd


def eval_match(gt_answer, gt_dtype, pred_answer, pred_dtype):
    if gt_dtype == pred_dtype:
        if isinstance(pred_answer, pd.Series):
            return pd.Series(eval(gt_answer)).equals(pred_answer)
        return gt_answer == pred_answer
    elif gt_dtype == "float" and pred_dtype == "int":
        return gt_answer == float(pred_answer)
    elif gt_dtype == "int" and pred_dtype == "float":
        return float(gt_answer) == pred_answer
    else:
        return False


if __name__ == "__main__":
    gt_ans = "{'West': '₹11,625.25', 'East': '₹2,617.50', 'South': '₹1,209.00', 'North': '₹600.00'}"
    pred_ans = pd.Series(["₹11,625.25", "₹2,617.50", "₹1,209.00", "₹600.00"], index=["West", "East", "South", "North"])
    print(eval_match(gt_ans, "pd.Series", pred_ans, "pd.Series"))
