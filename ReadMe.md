# Table Diversification: Evaluating LLM Robustness to Tabular Data Distortions

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

We investigate how large language models (LLMs) fail when tabular data in an otherwise canonical representation is subjected to semantic and structural distortions. Our findings reveal that LLMs lack an inherent ability to detect and correct subtle distortions in table representations. Only when provided with an explicit prior, via a system prompt, do models partially adjust their reasoning strategies and correct some distortions, though not consistently or completely.

To study this phenomenon, we introduce a small, expert-curated dataset that explicitly evaluates LLMs on table question answering (TQA) tasks requiring an additional error-correction step prior to analysis. Our results reveal systematic differences in how LLMs ingest and interpret tabular information under distortion, with even state-of-the-art models such as GPT-5.2 exhibiting a drop of minimum 22% accuracy under distortion. These findings raise important questions for future research, particularly regarding when and how models should autonomously decide to realign tabular inputs, analogous to human behavior, without relying on explicit prompts or tabular data pre-processing.

---

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Installation

### Prerequisites

- Python 3.12 or higher
- Docker (for code execution sandbox)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HNGM/table-diversification.git
   cd table-diversification
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r research/requirements.txt
   ```

3. **Docker Setup (Required for evaluation):**
   
   The agent under evaluation is given the power to make tool calls to execute Python scripts and upload Excel/CSV files on a Docker container.
   
   - Start the Docker service on your system:
     - [Windows Instructions](https://stackoverflow.com/a/44182489)
     - [Unix/Linux Instructions](https://docs.docker.com/engine/daemon/start/)
   
   - Build the Docker image:
     ```bash
     cd research/agents
     docker build -t tab-div-code-sandbox:latest .
     cd ../..
     ```

4. **Configure LLM API keys:**
   
   Set up your API keys in `config/default_llm_config.json` for the models you want to evaluate.

---

## Dataset

Our dataset consists of expert-curated table question answering tasks with two types of distortions:

### Dataset Structure

- **Location:** `research/dataset/overall_distorted_dataset/`
- **Files:**
  - `original.json` - Clean, canonical table representations
  - `disturbed.json` - Tables with semantic and structural distortions

### Data Files

All data files (Excel and image formats) are available in:
```
dev_test/Diversification/
├── Self Created Dataset/
│   └── Manual Created Diversified Dataset/
│       ├── Normal Diversifications/
│       └── Disturbed Diversifications/
└── External_Source/
    └── workbooks/
```

## Quick Start

### Running Evaluations

1. **Evaluate a model on the dataset:**
   ```bash
   python research/evaluation/prose_llm_main.py \
       --model dev-gpt-5-reasoning \
       --dataset overall_distorted_dataset \
       --data_mode disturbed \
       --prompt_mode default_mistake_no_sandbox \
       --ingest_mode markdown
   ```

2. **Analyze results:**
   ```bash
   python research/report/agg_score.py \
       --file research/results/<date>/<result_file>.json
   ```

---

## Experiments

### Evaluated Models

- **GPT Series:** GPT-5-Reasoning, GPT-5.1, GPT-5.2
- **Open Source:** DeepSeek-R1, Qwen-2.5-VL, Mistral-7B
- **Specialized:** TableGPT2-7B, TableLLM

### Evaluation Modes

1. **Ingest Modes:**
   - `markdown` - Tables as markdown format
   - `screenshot` - Visual table representations
   - `none` - Direct file upload

2. **Prompt Modes:**
   - `default` - Standard TQA instructions
   - `default_mistake` - Explicit distortion awareness prompts
   - `default_no_sandbox` - Without code execution
   - `default_tool_calling` - With function calling enabled

3. **Pass Rate:**
   - Multiple attempts per query (typically 3 passes)

### Running Custom Experiments

```bash
# Example: Evaluate GPT-5.2 with distortion awareness
python research/evaluation/prose_llm_main.py \
    --model dev-gpt-52-reasoning \
    --dataset overall_distorted_dataset \
    --data_mode disturbed \
    --prompt_mode default_mistake \
    --ingest_mode markdown \
    --pass_rate 3
```

---

## Results

All experimental results are stored in:
```
research/results/<date>/<experiment_config>.json
```

Format: `{dataset}_{datamode}_{promptmode}_{ingestmode}_{model}.json`


## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{table-diversification2026,
  title={Table Diversification: Evaluating LLM Robustness to Tabular Data Distortions},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please contact:
- **Email:** [avikdutta772000@gmail.com]
- **GitHub Issues:** [https://github.com/HNGM/table-diversification/issues](https://github.com/HNGM/table-diversification/issues)

---