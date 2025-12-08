import json
import os


final_json = []

with open("queries_and_answers.json", 'r') as fr:
    qa_list = json.load(fr)

div_types = ["Normal", "Disturbed"]

for file in qa_list:
    file_name = file[:-5]
    for div in div_types:
        for n, data_file in enumerate(os.listdir(f"Manual Created Dataset/{div} Diversifications/{file_name}")):
            for i, qa_pair in enumerate(qa_list[file]):
                final_json.append({
                    "index": f"{file_name}_{div}_{n}_{i}",
                    "query": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "type": div,
                    "data_file": [f"Manual Created Dataset/{div} Diversifications/{file_name}/{data_file}"],
                    "diversification_type": data_file.split(f"{file_name}_")[-1][:-5]
                })

with open("Manual_Created_Dataset.json", 'w') as fw:
    json.dump(final_json, fw, indent=4, ensure_ascii=False)
print(len(final_json))
