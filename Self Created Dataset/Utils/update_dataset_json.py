import json


file_name = "disturbed"
with open(f"../processed_dataset/original_datasets/{file_name}.json", "r") as fr:
    data = json.load(fr)

with open("reworked_queries_and_answers.json", "r", encoding="utf-8") as fqar:
    qa = json.load(fqar)

final_data = []
for sample in data:
    for qa_pairs in qa["_".join(sample["index"][:-4].split("_")[:-1])+".xlsx"]:
        if sample["query"] == qa_pairs["question"]:
            sample["answer"] = qa_pairs["answer"]
            sample["dtype"] = qa_pairs["dtype"]
            final_data.append(sample)
            break

with open(f"../processed_dataset/reworked_datasets/reworked_{file_name}.json", "w", encoding="utf-8") as fw:
    json.dump(final_data, fw, indent=4, ensure_ascii=False)
