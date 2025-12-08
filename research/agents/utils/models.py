from typing import List, Optional
from pydantic import BaseModel
from typing import List, Optional

class CodeOutput(BaseModel):
    type: str
    content: str

class ReturnedFile(BaseModel):
    download_link: str
    name: str
    path: str

class CodeRunResult(BaseModel):
    code_output_result: List[CodeOutput]
    deleted_files: List[ReturnedFile]
    new_generated_files: List[ReturnedFile]

class CodeRunData(BaseModel):
    is_partial: bool
    result: CodeRunResult

class RunCodeOutput(BaseModel):
    code: int
    message: str
    data: Optional[CodeRunData]
