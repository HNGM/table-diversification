import json
from src.utils.utils import ROOT_DIR
from pathlib import Path

PROMPT_DIR = ROOT_DIR / "research" / "agents" / "prompts"
PROMPT = {}

for prompt_file in PROMPT_DIR.glob("*.txt"):
    with open(prompt_file, "r", encoding="utf-8") as f:
        PROMPT[prompt_file.stem] = f.read()

def get_prompt(prompt_name: str) -> str:
    return PROMPT.get(prompt_name, "")
