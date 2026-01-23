# TrustingSyntheticData

Multiattribute Decision Analysis Framework for Synthetic Data Generation Method Selection

## Project Overview

This project implements a framework for evaluating and comparing synthetic data generation methods using multiattribute decision analysis. It helps researchers select the most appropriate synthetic data generation method based on their specific requirements for data quality, privacy, and utility.

## Tech Stack

- **Language**: Python 3.10+
- **Core Libraries**: pandas, numpy, scikit-learn, scipy
- **Synthetic Data**: SDV (CTGAN), DataSynthesizer (PrivBayes)
- **Testing**: pytest, pytest-cov
- **Formatting**: black, ruff

## Project Structure

```
TrustingSyntheticData/
├── src/                    # Source code modules
├── tests/                  # Test suite
├── data/                   # Data files
├── results/                # Generated results
├── notebooks/              # Jupyter notebooks
├── .agent/                 # Navigator documentation
└── pyproject.toml          # Project configuration
```

## Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov

# Format code
black .
ruff check --fix .
```

## Navigator Workflow

This project uses Navigator for documentation and context management.

### Session Start
- Run `/nav:start` to load the documentation navigator
- Navigator index: `.agent/DEVELOPMENT-README.md`

### Documentation Structure
- **Tasks**: `.agent/tasks/` - Implementation plans
- **System**: `.agent/system/` - Architecture docs
- **SOPs**: `.agent/sops/` - Standard procedures

### Key Commands
| Command | Purpose |
|---------|---------|
| `/nav:start` | Start session with navigator |
| `/nav:task` | Create/manage task docs |
| `/nav:compact` | Save context, clear for new work |

## Code Style

- Line length: 100 characters
- Use type hints where practical
- Follow PEP 8 conventions
- Run black and ruff before committing
