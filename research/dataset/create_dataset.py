import sys
sys.path.append(".")
from src.utils.utils import ROOT_DIR, read_json, write_json
from pathlib import Path

def main():
    dataset = read_json(r"C:\repo\ExcelCopilotResearch\TableDiversification\dev_test\Diversification\Self Created Dataset\Dataset JSONs\Manual_Created_Dataset(Combined).json")
    print(f"Loaded dataset with {len(dataset)} items.")
    output_path = ROOT_DIR / "research" / "dataset" / "processed_dataset"
    original = []
    diversified = []
    disturbed = []
    for data in dataset:
        data['data_file'] = [(ROOT_DIR / "dev_test" / "Diversification" / "Self Created Dataset" / path).relative_to(ROOT_DIR).as_posix() for path in data['data_file']]
        if data['type'] == "Normal":
            if data['data_file'][0].endswith("original.xlsx"):
                original.append(data)
            else:
                diversified.append(data)
        else:
            disturbed.append(data)
    write_json(original, output_path / "original.json")
    write_json(diversified, output_path / "diversified.json")
    write_json(disturbed, output_path / "disturbed.json")

def convert_to_string():
    dataset = ROOT_DIR / "research/dataset/processed_dataset/reworked_original.json"
    data = read_json(dataset)
    for item in data:
        item['answer'] = str(item['answer'])
    write_json(data, dataset)

def refactor_dataset():
    original = read_json(ROOT_DIR / "research" / "dataset" / "processed_dataset" / "original.json")
    div_dataset = []
    dist_dataset = []
    for data in original:
        data_file = data['data_file']
        orig_path = ROOT_DIR / Path(data_file)
        diversified_files = [f for f in orig_path.parent.iterdir() if f.is_file() and f != orig_path]
        disturbed_files = [f for f in (orig_path.parent.parent.parent / "Disturbed Diversifications" / orig_path.parent.name).iterdir() if f.is_file()]
        for div in diversified_files:
            new_data = data.copy()
            new_data['data_file'] = div.relative_to(ROOT_DIR).as_posix()
            new_data['mode'] = "diversified" 
            div_type = div.stem.replace(div.parent.stem + "_", "")
            new_data['diversification_type'] = div_type
            new_data['index'] = f"{data['index']}__{div_type}__diversified"
            div_dataset.append(new_data)
        for dist in disturbed_files:
            new_data = data.copy()
            new_data['data_file'] = dist.relative_to(ROOT_DIR).as_posix()
            new_data['mode'] = "disturbed" 
            dist_type = dist.stem.replace(dist.parent.stem + "_", "")
            new_data['diversification_type'] = dist_type
            new_data['index'] = f"{data['index']}__{dist_type}__disturbed"
            dist_dataset.append(new_data)
    write_json(div_dataset, ROOT_DIR / "research" / "dataset" / "processed_dataset" / "diversified.json")
    write_json(dist_dataset, ROOT_DIR / "research" / "dataset" / "processed_dataset" / "disturbed.json")

def add_screenshot_info():
    original = read_json(ROOT_DIR / "research" / "dataset" / "processed_dataset" / "original.json")
    diversified = read_json(ROOT_DIR / "research" / "dataset" / "processed_dataset" / "diversified.json")
    disturbed = read_json(ROOT_DIR / "research" / "dataset" / "processed_dataset" / "disturbed.json")
    for data in original:
        data_file = Path(data['data_file'])
        image_file = data_file.with_suffix('.png')
        data['image_file'] = image_file.as_posix()
    for data in diversified:
        data_file = Path(data['data_file'])
        image_file = data_file.with_suffix('.png')
        data['image_file'] = image_file.as_posix()
    for data in disturbed:
        data_file = Path(data['data_file'])
        image_file = data_file.with_suffix('.png')
        data['image_file'] = image_file.as_posix()
    write_json(original, ROOT_DIR / "research" / "dataset" / "processed_dataset" / "original.json")
    write_json(diversified, ROOT_DIR / "research" / "dataset" / "processed_dataset" / "diversified.json")
    write_json(disturbed, ROOT_DIR / "research" / "dataset" / "processed_dataset" / "disturbed.json")



def add_original():
    original = read_json(ROOT_DIR / "research" / "dataset" / "processed_dataset" / "reworked_original.json")
    for data in original:
        data['diversification_type'] = "original"
    write_json(original, ROOT_DIR / "research" / "dataset" / "processed_dataset" / "reworked_original.json")

def create_overall_distorted_dataset():
    prev = read_json(ROOT_DIR / "research" / "dataset" / "19122025_processed_dataset" / "original.json")
    curr = read_json(ROOT_DIR / "research" / "dataset" / "27122025_distorted_dataset" / "original.json")

    prev_dist = read_json(ROOT_DIR / "research" / "dataset" / "19122025_processed_dataset" / "disturbed.json")
    curr_dist = read_json(ROOT_DIR / "research" / "dataset" / "27122025_distorted_dataset" / "disturbed.json")

    dist_ann = read_json(ROOT_DIR / "research" / "report" / "analysis" / "disturbance_annotation.json")
    for dist in prev_dist:
        index = dist['index']
        annotation = [d for d in dist_ann if d['index'] == index]
        dist['distortion_type'] = annotation[0]['disturbance_annotation'] if annotation else "unknown"
    
    for dist in curr_dist:
        data_file = dist['data_file']
        if "semantic" in Path(data_file).name.lower():
            dist['distortion_type'] = "semantic"
        else:
            dist['distortion_type'] = "structural"

    combined = prev + curr
    combined_dist = prev_dist + curr_dist

    write_json(combined, ROOT_DIR / "research" / "dataset" / "overall_distorted_dataset" / "original.json")
    write_json(combined_dist, ROOT_DIR / "research" / "dataset" / "overall_distorted_dataset" / "disturbed.json")


if __name__ == "__main__":
    create_overall_distorted_dataset()