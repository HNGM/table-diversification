import json
from pathlib import Path


root_dir = Path(__file__).parent.parent.parent.parent.parent.resolve()
processed_datafile = root_dir/"research"/"results"/"171225"/"original.json"

with open(processed_datafile, 'r') as fr:
    dataset = json.load(fr)

# ['index', 'query', 'answer', 'dtype', 'type'], ['data_file', 'image_file', 'diversification_type', 'eval']
final_data = []
for sample in dataset:
    del sample['eval']
    del sample['data_file']
    del sample['image_file']
    del sample['diversification_type']
    final_data.append(sample)

with open("updated_queries_and_answers.json", 'w') as fw:
    json.dump(final_data, fw, indent=4)
