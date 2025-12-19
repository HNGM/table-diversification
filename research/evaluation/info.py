import json
from pathlib import Path
from typing import Any, List, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, PlainSerializer, model_validator
from src.utils.utils import read_file, write_json

PosixPath = Annotated[
    Path, PlainSerializer(lambda x: x.as_posix(), return_type=str, when_used='json')
]

class Info(BaseModel):
    index: str
    query: str
    answer: Any
    data_file: PosixPath
    image_file: Optional[PosixPath] = None
    diversification_type: str
    dtype: str
    type: Optional[str] = None

    @classmethod
    def get_info(cls, path: Path)->List['Info']:
        infos = read_file(path)
        return [cls(**info) for info in infos]