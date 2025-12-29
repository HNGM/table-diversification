import json
from os import listdir
from pathlib import Path
import copy


root_dir = Path(__file__).parent.parent.parent.resolve()
dataset_dir = "dev_test/Diversification/External_Source/workbooks"

with open(root_dir/"dev_test/Diversification/External_Source/queries_and_answer.json", 'r') as fr:
    qa_pairs = json.load(fr)

original = []
disturbed = []
for qa in qa_pairs:
    workbook_name = qa["index"][:-2]
    query_number = qa["index"][-1]
    og_temp = copy.deepcopy(qa)
    og_temp["data_file"] = f"{dataset_dir}/{workbook_name}/{workbook_name}.xlsx"
    og_temp["image_file"] = f"{dataset_dir}/{workbook_name}/{workbook_name}.png"
    og_temp["diversification_type"] = "original"
    original.append(og_temp)

    for file in listdir(root_dir/dataset_dir/workbook_name/"disturbed"):
        file_path = f"{dataset_dir}/{workbook_name}/disturbed/{file}".replace("\\", "/")
        if query_number in file:
            dis_temp = copy.deepcopy(qa)
            dis_temp["data_file"] = file_path
            dis_temp["image_file"] = file_path[:-5]+".png"
            dis_temp["diversification_type"] = "disturbed"
            disturbed.append(dis_temp)

with open("27122025_distorted_dataset/original.json", 'w') as fw:
    json.dump(original, fw, indent=4)

with open("27122025_distorted_dataset/disturbed.json", 'w') as fw:
    json.dump(disturbed, fw, indent=4)
