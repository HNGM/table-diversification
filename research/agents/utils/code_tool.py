import os
import pathlib
import time
import re
from pathlib import Path
from hashlib import sha256
from typing import List, Optional, Union
try:
    import docker
except ImportError:
    raise ImportError("Please install the docker package via 'pip install docker'")

class CodeToolRequest:
    def __init__(self, code_str: str):
        code_blocks = re.findall(r'```(?:python)?(?:py)?\s*(.*?)\s*```', code_str, re.DOTALL)
        
        if len(code_blocks) == 0:
            self.code = code_str.strip()
        else:
            self.code = '\n'.join(code_blocks).strip()
        lines = self.code.splitlines(keepends=True)
        if lines and "print" not in lines[-1]:
            last_line = lines[-1]
            last_line_lstrip = last_line.lstrip()
            indent = last_line[:last_line.index(last_line_lstrip)]
            line_content = last_line_lstrip.strip()
            if line_content.startswith("#"):
                return
            lines[-1] = indent + f"print({line_content})" + last_line.split(line_content, 2)[1]
            self.code = "".join(lines)
        
class CodeToolResponse:
    def __init__(self, exit_code: int, log: str, output_files: Optional[list[Path]]):
        self.exit_code = exit_code
        self.log = log
        self.output_files = output_files

class CodeTool:
    def __init__(
        self,
        image: str = "tab-div-code-sandbox:latest",
        time_out: int = 150,
        work_dir: str = "./.tmp_code",
        output_dir: str = "./.tmp_output",
        sandbox_id: Optional[str] = None,
    ):
        self._client = docker.from_env()
        self._image = image
        self._time_out = time_out
        if sandbox_id is None:
            self.sandbox_id = sha256(str(time.time()).encode()).hexdigest()
        self._work_dir = pathlib.Path(work_dir) / self.sandbox_id
        self._output_dir = pathlib.Path(output_dir) / self.sandbox_id
        self._code_idx = sha256(str(time.time()).encode()).hexdigest()
        self._log_len = 0
        os.makedirs(self._work_dir, exist_ok=True)
        os.makedirs(self._output_dir, exist_ok=True)
        
        self._uploaded_files = []
        self._generated_files = []
        
        self._description = """
        Python Code Execution Sandbox. The Sandbox just executes a given python code and returns the stdout and stderr 
        of the executed code. Note that the sandbox just returns the code output and nothing else. 
        
        **IMPORTANT** Your response should always contain a single and complete python program. Do not write code snippets.
        
        It also contains the following uploaded files that can be used in the code:
        """
    
    @property
    def description(self):
        return self._description + "\n".join([f"{i}. {file.name}" for i, file in enumerate(self._uploaded_files)])
    def get_code_filename(self):
        code_idx = sha256(str(time.time()).encode()).hexdigest()
        return f"exec_code_{code_idx}.py"
        
    def run_code(self, code_str: str) -> CodeToolResponse:
        req = CodeToolRequest(code_str)
        if not req.code:
            return CodeToolResponse(1, "No code to execute", "")

        file_name = self.get_code_filename()
        file_path = os.path.join(self._work_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.code)

        cmd = f"python /workspace/{file_name}"
        container = self._client.containers.run(
            image=self._image,
            command=cmd,
            detach=True,
            working_dir="/output",
            volumes={
                pathlib.Path(self._work_dir).resolve(): {
                    "bind": "/workspace",
                    "mode": "rw",
                },
                pathlib.Path(self._output_dir).resolve(): {
                    "bind": "/output",
                    "mode": "rw",
                }
            },
            environment={
                "OUTPUT_DIR": "/output"
            }
        )

        start_time = time.time()
        while container.status != "exited" and (time.time() - start_time < self._time_out):
            container.reload()

        if container.status != "exited":
            container.stop()
            container.remove()
            return CodeToolResponse(1, "TIMEOUT", "")

        logs = container.logs().decode("utf-8").rstrip()
        exit_code = container.attrs["State"]["ExitCode"]
        container.remove()

        return CodeToolResponse(exit_code, logs, self.register_generated_files())

    def register_generated_files(self):
        """
        Register newly generated files in the output directory in self._generated_files
        Return the list of newly generated files.
        """
        current_files = set(Path(self._output_dir).glob('*'))
        new_files = [f for f in current_files if f not in self._generated_files and str(f) not in self._uploaded_files]
        self._generated_files.extend(new_files)
        return new_files

    def download_generated_images(self, save_dir: Union[str, Path]) -> List[Path]:
        """Download generated images to specified directory.
        
        Args:
            save_dir: Directory to save images to
        Returns:
            List of paths to downloaded images
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        for file in self._generated_files:
            if file.suffix in [".png", ".jpg", ".jpeg"]:
                save_file_path = save_path / file.name
                content = file.read_bytes()
                    
                save_file_path.write_bytes(content)
                downloaded.append(save_file_path)
            
        return downloaded
    
    def upload_files(self, file_paths: List[Path]) -> List[Path]:
        """
        Upload files to work_dir by copying them
        """
        copied_files = []
        for file_path in file_paths:
            if not file_path.exists():
                print(f"File {file_path} does not exist")
                continue
            
            dest_path = Path(self._output_dir) / file_path.name
            with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            
            copied_files.append(dest_path)
            self._uploaded_files.append(dest_path)
        
        return copied_files


if __name__ == "__main__":
    # Basic sanity check
    code_tool = CodeTool(time_out=5)  # shorter timeout for testing
    
    # Test 1: Basic code execution
    print("Test 1: Basic code execution")
    response = code_tool.run_code("```print('Hello, World!')```")
    print(response.log)
    #Test 2: Upload Files
    print("Test 2: Upload Files")
    paths = [path.name for path in code_tool.upload_files([Path('ada_agents/code_tool.py')])]
    response = code_tool.run_code(f"```import pandas\nfrom matplotlib.pyplot as plt\nfor path in {paths}:\n\tprint(open(path).read())```")
    print(response.log)
    print(response.output_files)
