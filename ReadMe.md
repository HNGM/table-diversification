# Table Diversification

A research project for evaluating data analysis agents on diversified table structures. This project tests AI agents' ability to answer analytical queries across various table formats (hierarchical headers, transposed tables, merged columns, etc.) to measure robustness to structural variations.

## Project Structure

### Root Directory

- **`ReadMe.md`**: Project documentation
- **`config/`**: Configuration files
  - `default_llm_config.json`: LLM endpoint configurations (Azure ML, Gemini, etc.)

### `src/` - Core Framework

Core utilities and interfaces for building and running agents.

- **`interfaces/`**: Abstract agent interfaces and message handling
  - `agent.py`: Base `Agent` class with response handling
  - `ada_agent.py`: Agent with file upload and attachment capabilities
  - `chat_agent.py`: Conversational agent with tool/function calling support
  - `message.py`: Message classes (`UserMessage`, `Message`)
  - `workflow.py`: Workflow orchestration framework
  
- **`utils/`**: Shared utilities
  - `data_models.py`: Data models for the project
  - `data_preview.py`: Data preview utilities
  - `llm_config.py`: LLM configuration management
  - `llm_response_parser.py`: Response parsing utilities
  - `logger.py`: Logging utilities
  - `substrate.py`: Substrate/backend utilities
  - `utils.py`: General utility functions

### `research/` - Evaluation Pipeline

Research code for dataset creation, agent implementation, and evaluation.

- **`agents/`**: Agent implementations
  - `agent.py`: `FunctionCallAdaAgent` - Main agent with code execution capabilities
  - `output_format.py`: Response format specifications
  - `Dockerfile`: Docker setup for sandboxed code execution
  - `prompts/`: Agent system prompts
    - `default.txt`: Default agent instructions
  - `utils/`: Agent utilities
    - `code_tool.py`: Python code execution sandbox in Docker
    - `model_response.py`: Model response handling
    - `models.py`: Model definitions

- **`dataset/`**: Dataset generation and processing
  - `create_dataset.py`: Script to organize datasets by type (original/diversified/disturbed)
  - `processed_dataset/`: Organized datasets in JSON format
    - `original.json`: Queries on original table structures
    - `diversified.json`: Queries on diversified structures (hierarchical headers, transposed, etc.)
    - `disturbed.json`: Queries on disturbed structures
    - `reworked_*.json`: Refined versions of datasets

- **`evaluation/`**: Evaluation framework
  - `main.py`: Entry point for running evaluations
  - `evaluator.py`: `QualitativeEvaluator` - LLM-based response evaluator
  - `llm_evaluator.py`: LLM evaluation logic
  - `evaluate.py`: Evaluation orchestration
  - `info.py`: Information models for evaluation
  - `utils.py`: Evaluation utilities

- **`report/`**: Analysis and reporting
  - `agg_score.py`: Aggregate score calculation from evaluation results

- **`results/`**: Evaluation results organized by date
  - Format: `DDMMYY/` containing evaluated JSON files with verdicts

- **`Readme.md`**: Setup instructions (Docker, environment)
- **`requirements.txt`**: Python dependencies

### `dev_test/` - Development and Testing

Development datasets and experimental evaluations.

- **`Diversification/Self Created Dataset/`**: Manually created test datasets
  
  - **`Dataset JSONs/`**: Combined dataset definitions
    - `Manual_Created_Dataset(Combined).json`: All queries combined
    - `Manual_Created_Normal_Dataset.json`: Normal table queries
    - `Manual_Created_Disturbed_Dataset.json`: Disturbed table queries
  
  - **`Evaluation/`**: Evaluation results by date (format: `MMDDYYYY/`)
    - Contains evaluated results with naming pattern: `{Type}_Eval_toolcall_Manual_Created_Dataset.json`
    - Types include: Normal, Disturbed, Prompted variants with different percentages
  
  - **`Manual Created Dataset/`**: Source Excel files
    - `Disturbed Diversifications/`: Tables with structural disturbances
    - `Normal Diversifications/`: Tables with standard diversifications
    - Organized by domain: `Retail_Sales_Orders/`, `HR_Timesheets/`, `Logistics_Shipments/`, `Clinic_Visits/`, `Personal_Finance_Transactions/`, `University_Grades/`
  
  - **`Original Files/`**: Original undiversified table files
  
  - **`Simulation/`**: Agent simulation runs organized by date
    - Contains intermediate results during evaluation runs
  
  - **`simul_plot_dir/`**: Generated plots from simulations
  
  - **`Utils/`**: Dataset generation utilities
    - `generate_dataset.py`: Main dataset generator (creates 6 domain Excel files)
    - `generate_json_for_manual_data.py`: Converts manual datasets to JSON format
    - `create_data_excel.py`: Excel creation utilities
    - `queries_and_answers.json`: Query-answer pairs for evaluation

## Key Concepts

### Table Diversification Types

1. **Original**: Standard table format
2. **Normal Diversification**: Structural variations without data corruption
   - Hierarchical column/row headers
   - Transposed tables
   - Merged similar columns
   - Single hierarchical headers
3. **Disturbed**: Tables with intentional structural challenges to test robustness

### Evaluation Flow

1. **Dataset Creation**: Generate Excel files with various table structures
2. **Agent Execution**: Run agent on queries using `FunctionCallAdaAgent`
3. **Qualitative Evaluation**: Use `QualitativeEvaluator` to compare agent responses against reference answers
4. **Aggregation**: Calculate success rates using `agg_score.py`

### Docker Sandbox

The agent executes Python code in a Docker container (`tab-div-code-sandbox:latest`) for security and isolation, allowing safe file manipulation and data analysis operations.

## Setup

See `research/Readme.md` for detailed setup instructions including Docker configuration.

## Usage

Main evaluation entry point:
```bash
python research/evaluation/main.py [arguments]
```

Generate aggregate scores:
```bash
python research/report/agg_score.py
```