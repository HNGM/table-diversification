import json
from os import listdir
from pathlib import Path
import copy


root_dir = Path(__file__).parent.parent.parent.resolve()
dataset_dir = "dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset"

with open(root_dir/"dev_test/Diversification/Self Created Dataset/Utils/updated_queries_and_answers.json", 'r') as fr:
    qa_pairs = json.load(fr)

div_types = ["Normal", "Disturbed"]
original = []
diversified = []
disturbed = []
for div_type in div_types:
    for og_file in listdir(root_dir/dataset_dir/f"{div_type} Diversifications"):
        for file in listdir(root_dir/dataset_dir/f"{div_type} Diversifications"/og_file):
            div_applied = file[len(og_file)+1:-5]
            file_path = f"{dataset_dir}/{div_type} Diversifications/{og_file}/{file}".replace("\\", "/")
            if div_type == "Normal":
                if "original" in file:
                    for qa in qa_pairs:
                        if og_file in qa["index"]:
                            temp = copy.deepcopy(qa)
                            temp["data_file"] = file_path
                            temp["image_file"] = file_path[:-5]+".png"
                            temp["diversification_type"] = div_applied
                            original.append(temp)
                else:
                    for qa in qa_pairs:
                        if og_file in qa["index"]:
                            temp = copy.deepcopy(qa)
                            temp["index"] = temp["index"]+"__"+div_applied+"__diversified"
                            temp["data_file"] = file_path
                            temp["image_file"] = file_path[:-5]+".png"
                            temp["diversification_type"] = div_applied
                            diversified.append(temp)
            else:
                for qa in qa_pairs:
                    if og_file in qa["index"]:
                        temp = copy.deepcopy(qa)
                        temp["index"] = temp["index"]+"__"+div_applied+"__disturbed"
                        temp["data_file"] = file_path
                        temp["image_file"] = file_path[:-5]+".png"
                        temp["diversification_type"] = div_applied+"_disturbed"
                        disturbed.append(temp)

# print(*original, sep="\n", end="\n\n")
# print(*diversified, sep="\n", end="\n\n")
# print(*disturbed, sep="\n")

with open("19122025_processed_dataset/original.json", 'w') as fw:
    json.dump(original, fw, indent=4)

with open("19122025_processed_dataset/diversified.json", 'w') as fw:
    json.dump(diversified, fw, indent=4)

with open("19122025_processed_dataset/disturbed.json", 'w') as fw:
    json.dump(disturbed, fw, indent=4)
