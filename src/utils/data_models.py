import json
from pathlib import Path
from typing import Any, List, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, PlainSerializer, model_validator
from .utils import read_file, write_json

PosixPath = Annotated[
    Path, PlainSerializer(lambda x: x.as_posix(), return_type=str, when_used='json')
]


class MinedInfo(BaseModel):
    dataset_files : List[PosixPath]
    image_files : Optional[List[PosixPath]] = []
    article : Optional[str] = ""
    article_uri : Optional[str] = ""
    source_uri : Optional[str] = ""
    date : Optional[str] = ""
    article_index : str
    readme_path : Optional[str] = ""

    @classmethod
    def get_info(cls, path: Path)->List['MinedInfo']:
        mined_infos = read_file(path)
        return [cls(**mined_infos) for mined_infos in mined_infos]

class CuratedInfo(BaseModel):
    image_files : Optional[List[PosixPath]] = []
    query : str
    answer : str
    type : str
    data_file : List[PosixPath]
    index : str

    @classmethod
    def get_info(cls, path: Path)->List['CuratedInfo']:
        mined_infos = read_file(path)

        curated_infos = []
        for curated in mined_infos:
            for i, qa in enumerate(curated['query_answer']['qa_list']):
                flatten_dict = {cur_keys: cur_values for cur_keys, cur_values in curated.items() if cur_keys != "query_answer"}
                if not isinstance(qa, dict):
                    continue
                flatten_dict.update(qa)
                flatten_dict['index'] = flatten_dict["article_index"] + "_" + flatten_dict["type"] + "_" + flatten_dict["query_index"]

                try:
                    curated_infos.append(cls(**flatten_dict))
                except:
                    continue
        return curated_infos
    
    
    @model_validator(mode='before')
    @classmethod
    def validate_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'data_file' not in data:
                data['data_file'] = data.get('dataset_files', [])
            elif isinstance(data['data_file'], str):
                data['data_file'] = [data['data_file']]
        return data


class CodeGeneratedInfo(BaseModel):
    image_files : Optional[List[PosixPath]] = []
    query : str
    answer : str
    type : str
    data_file : List[PosixPath]
    index : str
    codegen_result : str = None
    codegen_interaction_log : Optional[dict] = {}
    program : str
    program_output : Union[None, str] = None
    program_plot : Union[PosixPath, None] = None

    @classmethod
    def get_info(cls, path: Path)->List['CodeGeneratedInfo']:
        mined_infos = read_file(path)
        codegen_infos = []
        for codegen_info in mined_infos:
            codegen_dict = codegen_info.pop('codegen')
            if not codegen_dict:
                continue
            elif not codegen_dict.get('codegen_result', None):
                continue
            codegen_info.update(codegen_dict)
            codegen_infos.append(cls(**codegen_info))

        return codegen_infos

class BenchmarkInfo(BaseModel):
    index : str
    query : str
    answer : str
    type : str
    data_file : List[PosixPath]
    program : str
    program_output : Union[None, str] = None
    program_plot : Union[PosixPath, None] = None

    @classmethod
    def get_info(cls, path: Path)->List['BenchmarkInfo']:
        benchmark_infos = read_file(path)
        return [cls(**benchmark_info) for benchmark_info in benchmark_infos]

    @classmethod
    def convert_to_benchmark(cls, input_file:Path, output_file:Path):
        codegen_infos = CodeGeneratedInfo.get_info(input_file)
        benchmark_data = []
        for codegen_info in codegen_infos:
            if codegen_info.codegen_result == "success":
                benchmark_data.append(cls(**vars(codegen_info)).model_dump(mode="json"))
                write_json(benchmark_data, output_file)


class SimulationInfo(BaseModel):
    index : Union[str, int]
    query : str
    answer : str
    type : str
    data_file : List[PosixPath]
    program : str
    program_output : Union[str, None] = None
    program_plot : Union[PosixPath, None] = None
    tool_response : str
    tool_plot_path : List[PosixPath] = []
    conv_log : List[dict] = []

    @classmethod
    def get_info(cls, path: Path)->List['SimulationInfo']:
        simulation_infos = read_file(path)
        return [cls(**simulation_info) for simulation_info in simulation_infos]



        
