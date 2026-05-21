import json
import os
from Distortion_Generator import gen_distortions
from Distortion_Selector import run_selector
from Distortion_Reviewer import run_reviewer
from pathlib import Path


DISTORTION_CATALOGUE = {
    # Structural distortions – alter table layout, row/column structure
    "merge_cells": "Structural",
    "horizontal_shift": "Structural",
    "vertical_shift": "Structural",
    "broken_rows_split": "Structural",
    "broken_rows_merge": "Structural",
    "header_as_data_row": "Structural",
    "multi_column_collapse": "Structural",
    "footnote_injection": "Structural",
    # Semantic distortions – alter cell content, formatting, or meaning
    "ocr_char_misinterpret": "Semantic",
    "ocr_lost_formatting": "Semantic",
    "numbers_as_text": "Semantic",
    "date_format_corruption": "Semantic",
    "decimal_separator_swap": "Semantic",
    "random_noise_chars": "Semantic",
    "context_loss": "Semantic"}


def Run_Generator():

    with open("JSONs/Dev_500/500_Dev_Samples.json", "r", encoding="utf-8") as fr:
        data = json.load(fr)

    for n, sample in enumerate(data):
        file_path = Path(sample["data_file"])
        folder, file = file_path.parent.name, file_path.stem
        print(f"{n}. Processing {folder}/{file}{file_path.suffix}...")
        os.makedirs(f"Output/05082026/{folder}/{file}/{sample['index']}", exist_ok=True)
        gen_distortions(input_path=sample["data_file"],
                        output_path=f"Output/05082026/{folder}/{file}/{sample['index']}/{file}_{sample['index']}.xlsx",
                        distortion=sample["suitable_distortions"],
                        question=sample["query"], answer=sample["answer"],
                        individual_workbook=True)


def Create_Generator_JSON():

    with open("JSONs/Dev_500/500_Dev_Samples.json", "r", encoding="utf-8") as fr:
        data = json.load(fr)

    generator_data = []
    for n, sample in enumerate(data):
        file_path = Path(sample["data_file"])
        folder, file = file_path.parent.name, file_path.stem
        print(f"{n}. Processing Output/05082026/{folder}/{file}/{sample['index']}...")
        for distortion in sample["suitable_distortions"]:
            generator_data.append({
                "index": f"{sample['index']}_{distortion}",
                "query": sample["query"],
                "answer": sample["answer"],
                "dtype": sample["dtype"],
                "original_file": f"Output/05082026/{folder}/{file}/{sample['index']}/{file}_{sample['index']}_original.xlsx",
                "data_file": f"Output/05082026/{folder}/{file}/{sample['index']}/{file}_{sample['index']}_{distortion}.xlsx",
                "distortion": distortion,
                "distortion_type" : DISTORTION_CATALOGUE[distortion]
            })

    with open("JSONs/Dev_500/500_Dev_Distorted_Samples.json", "w", encoding="utf-8") as fw:
        json.dump(generator_data, fw, ensure_ascii=False, indent=4)


def Run_Selector(start_from_backup=False):

    with open("JSONs/Dev_500/dev_split_500.json", "r", encoding="utf-8") as fr:
        data = json.load(fr)

    if start_from_backup:
        with open("JSONs/Dev_500/backups/500_Dev_Samples_backup.json", "r", encoding="utf-8") as fr:
            reviewed_data = json.load(fr)
        print("Samples processed:", len(reviewed_data))
    else:
        reviewed_data = []

    for n, sample in enumerate(data[len(reviewed_data):], start=len(reviewed_data)):
        print(f"{n}. Processing {sample["data_file"]}...")
        review = run_selector(filepath=sample["data_file"], question=sample["query"],
                              answer=sample["answer"], model="gpt-5.2")

        sample["suitable_distortions"] = review["recommended"]
        reviewed_data.append(sample)

        if n%10 == 0:
            with open("JSONs/Dev_500/backups/500_Dev_Samples_backup.json", "w", encoding="utf-8") as fbw:
                json.dump(reviewed_data, fbw, ensure_ascii=False, indent=4)

    with open("JSONs/Dev_500/500_Dev_Samples.json", "w", encoding="utf-8") as fw:
        json.dump(reviewed_data, fw, ensure_ascii=False, indent=4)


def Run_Reviewer():
    run_reviewer(samples_json="JSONs/Dev_500/500_Dev_Distorted_Samples.json",
                 output_json="JSONs/Dev_500/500_Dev_Distorted_Samples_Reviewed.json",
                 stats_json="JSONs/Dev_500/500_Dev_Review_Stats.json",
                 verbose=True, model="gpt-5.2", limit=None)


if __name__ == "__main__":
    # Run_Selector(start_from_backup=False)  # 1
    # Run_Generator()  # 2
    # Create_Generator_JSON()  # 3
    Run_Reviewer()  # 4
