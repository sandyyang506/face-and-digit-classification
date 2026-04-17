# CS 440 Final Project - Face and Digit Classification

## Team Members
- Neil Gandhi
- Dev Patel
- Sandy Yang

## Setup Instructions

### Prerequisites
Install `uv`:
- Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Installation
```bash
git clone <repo-url>
cd face-and-digit-classification
uv sync
curl -o data/data.zip http://rl.cs.rutgers.edu fall2019/data.zip
unzip data/data.zip -d data/
```

### Run
```bash
uv run python src/main.py
```

## Project Structure
```
```