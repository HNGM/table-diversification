from typing import Dict, List, Tuple, Union
from pathlib import Path
import pandas as pd
import json

root_dir = Path(__file__).parent.resolve()
while root_dir.name != "TableDiversification":
    root_dir = root_dir.parent
ROOT_DIR = root_dir

def read_json(filepath: Union[Path, str]):
    assert Path(filepath).exists(), "JSON does not exist!"
    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
        json_data = json.load(file)
    return json_data


def write_json(json_data, filepath: Union[Path, str]):
    with open(filepath, "w") as file:
        json.dump(json_data, file, indent=2)

def read_as_dataframe(filepath: Union[Path, str])->Dict[str, pd.DataFrame]:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if filepath.suffix == '.csv':
        try:
            encoding = "utf-8"
            sheet_df_dict = {filepath.stem: pd.read_csv(filepath, encoding=encoding, low_memory=False)}
        except UnicodeDecodeError:
            try:
                encoding = "cp1252"
                sheet_df_dict = {filepath.stem: pd.read_csv(filepath, encoding=encoding, low_memory=False)}
            except UnicodeDecodeError:
                encoding = "latin1"
                sheet_df_dict = {filepath.stem: pd.read_csv(filepath, encoding=encoding, low_memory=False)}
    elif filepath.suffix == '.xlsx':
        sheet_df_dict = pd.read_excel(filepath, sheet_name=None)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    return sheet_df_dict

def write_from_dataframe(df: pd.DataFrame, filepath: Union[Path, str]):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if filepath.suffix == '.csv':
        df.to_csv(filepath, index=False)
    elif filepath.suffix == '.xlsx':
        with pd.ExcelWriter(filepath, mode='w', engine="xlsxwriter", engine_kwargs={'options': {'strings_to_urls': False}}) as writer:
            df.to_excel(writer, index=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def write_df_to_multiple_sheets(sheet_df: List[Tuple[str, pd.DataFrame]], filepath: Union[Path, str]):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if filepath.suffix == '.xlsx':
        with pd.ExcelWriter(filepath) as writer:
            for sheet_name, df in sheet_df:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}\nRequires '.xlsx' format for workbook.")


def write_multiple_df_to_sheet(sheet_df: List[Tuple[str, pd.DataFrame]], filepath: Union[Path, str], alignment: str):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if alignment not in ["horizontal", "vertical"]:
        raise ValueError(f"Unsupported alignment: {alignment}")

    start_row = 0
    start_col = 0
    if filepath.suffix == '.xlsx':
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            for sheet_name, df in sheet_df:
                sn = sheet_name.split("_")[0]
                df.to_excel(writer, index=False, startrow=start_row, startcol=start_col, sheet_name=sn)
                if alignment == "horizontal":
                    start_col += len(df.columns) + 1
                elif alignment == "vertical":
                    start_row += len(df.index) + 2
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}\nRequires '.xlsx' format.")


def read_jsonl(filename: Union[str, Path]):
    data_list = []
    with open(filename, "r") as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)
    return data_list


def read_file(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == '.json':
        mined_infos = read_json(path)
    elif path.suffix == '.jsonl':
        mined_infos = read_jsonl(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return mined_infos
    
