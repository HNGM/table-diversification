import json
from pprint import pprint
import pandas as pd


types = ["", "Div_", "Disturbed_"]
num = 0

with open(f"{types[num]}Manual_Created_Dataset.json", 'r') as fr:
    all_samples = json.load(fr)

with open(f"{types[num]}Eval_toolcall_Manual_Created_Dataset.json", 'r') as fr:
    div_eval_data = json.load(fr)

divs = ['transpose_table', 'single_hierarchical_row_header', 'merge_similar_columns', 'hierarchical_column_header', "original"]

data_dict_verdict = {
    "Index": [],
    "original": [],
    "transpose_table": [],
    "single_hierarchical_row_header": [],
    "merge_similar_columns": [],
    "hierarchical_column_header": []
}

data_dict_response = {
    "Index": [],
    "original": [],
    "transpose_table": [],
    "single_hierarchical_row_header": [],
    "merge_similar_columns": [],
    "hierarchical_column_header": []
}

for sample in all_samples:
    index = sample['index'][:-4]
    data_dict_verdict['Index'].append(index)
    data_dict_response['Index'].append(index)
    all_divs = {'transpose_table', 'single_hierarchical_row_header', 'merge_similar_columns', 'hierarchical_column_header', "original"}

    for div_sample in div_eval_data:
        if index in div_sample['index']:
            for div in divs:
                if div == sample["diversification_type"]:
                    data_dict_verdict[div].append(div_sample["qualitative"]["verdict"])
                    data_dict_response[div].append(div_sample["tool_response"])
                    all_divs.remove(div)

    for div in all_divs:
        data_dict_verdict[div].append(None)
        data_dict_response[div].append(None)

# Convert the dictionary to a DataFrame
df_verdict = pd.DataFrame(data_dict_verdict)
df_response = pd.DataFrame(data_dict_response)
# print(df)

with pd.ExcelWriter(f"{types[num]}Eval_toolcall_Manual_Created_Dataset.xlsx") as writer:
    df_verdict.to_excel(writer, sheet_name="verdict", index=False)
    df_response.to_excel(writer, sheet_name="response", index=False)
