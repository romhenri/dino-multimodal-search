# RUN.md

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run
```bash
# :8000 (first start downloads E5 + CLIP L/14, ~2.8 GB)
uvicorn app.main:app
```

## Test / eval
```bash
pytest
python -m eval.run_eval
python -m eval.run_sketch_eval eval/gt/*.json
```
