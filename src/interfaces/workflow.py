import datetime
import multiprocessing
from abc import ABC, abstractmethod
import json
from pathlib import Path
import random
import sys
import traceback
from typing import Optional, Union, Tuple
from multiprocessing import Pool, Event
from tqdm import tqdm
import signal

from ..utils.llm_config import LLMConfig, load_llm_configs
from ..utils.utils import write_json

class Workflow(ABC):
    def __init__(
        self,
        llm_config_path: Union[Path, str],
        input_file: Union[Path, str],
        output_file: Union[Path, str],
        model: str,
        name: str = "Generic",
        nproc: int = 5,
        resume: bool = False,
        **kwargs
    ):
        self.llm_config_path = Path(llm_config_path)
        self.llm_configs = load_llm_configs(llm_config_path, model)
        print(f"Loaded {len(self.llm_configs)} LLMConfigs")
        print(f"LLMConfigs: {self.llm_configs}")
        self.input_file = input_file
        self.output_file = output_file
        self.name = name
        self.nproc = nproc
        self.resume = resume

    def run(self):
        return self._run()
    
    def _process_single_item(self, args) -> Tuple[Optional[dict], Optional[str]]:
        i, input_data = args

        # Use first config (client is cached, so safe for concurrent access)
        config = self.llm_configs[i % len(self.llm_configs)]
        
        try:
            result = self.workflow(config, input_data)
            return result, None, None
        except Exception as e:
            error_msg = f"\nError processing item {i}:\n{str(e)}\n{traceback.format_exc()}\n{'-' * 80}"
            return None, error_msg, e

    def _filter_already_processed_indices(self, input_dataset):
        if not self.output_file.exists():
            raise Exception("Output file does not exist. Cannot resume from a non-existent output file.")
        proccessed_data =  [item for item in json.load(self.output_file.open())]
        proc_set = set([item["index"] for item in proccessed_data])

        init_size = len(input_dataset)

        input_dataset = [item for item in input_dataset if item.index not in proc_set]
        print(f"Resuming from the last checkpoint. {init_size - len(input_dataset)} items have already been processed.")
        random.shuffle(input_dataset)
        return input_dataset, proccessed_data


    def _run(self):
        if self.output_file.exists() and not self.resume:
            raise Exception(f"Output file {self.output_file} already exists. Delete this or specify a different output file.")
        input_dataset = self.load_data()
        results = []
        successful_output = []
        shutdown = Event()

        if self.output_file.exists():
            print(f"Output file {self.output_file} already exists. Will resume from the last checkpoint.")
            input_dataset, successful_output = self._filter_already_processed_indices(input_dataset)

        print(f"Running {self.name} workflow in parallel with {self.nproc} processes...")

        try:
            with Pool(processes=self.nproc, initializer=init_worker) as pool:
                iterator = pool.imap_unordered(
                    self._process_single_item,
                    enumerate(input_dataset)
                )
                
                for result in tqdm(
                    iterator,
                    total=len(input_dataset),
                    desc="Processing items"
                ):
                    if shutdown.is_set():
                        break
                        
                    if result[2] is not None:
                        print(result[1])
                        if isinstance(result[2], KeyboardInterrupt):
                            shutdown.set()
                            break
                    else:
                        if result[0] is not None:
                            successful_output.append(result[0])
                            write_json(successful_output, self.output_file)
                    results.append(result)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, shutting down...")
            shutdown.set()
            return successful_output
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return successful_output
            
        finally:
            if 'pool' in locals():
                pool.close()
                pool.join()

        if not results:
            print("No items were processed.")
            
        print(f"Completed {self.name} workflow with {len(results)} items processed.")
        print(f"Total successful runs: {len([r for r in results if r[2] is None])}")
        print(f"Total errors encountered: {len([r for r in results if r[2] is not None])}")
        print(f"Input dataset size: {len(input_dataset)}")
        print(f"Output dataset size: {len(successful_output)}")
        
        return successful_output

    @abstractmethod
    def workflow(self, llm_config: LLMConfig, *args, **kwargs):
        pass
    
    @abstractmethod
    def load_data(self):
       pass

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)