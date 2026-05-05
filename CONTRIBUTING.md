# Contributing to ARN

Thanks for helping improve ARN.

## Local dev setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest
```

## Pull request checklist

- Keep the core memory behavior working: store facts, recall by meaning, persist across sessions.
- Add or update tests for changes.
- Do not add network calls to runtime memory operations.
- Keep OpenClaw integration optional.
- Run `arn selftest --strict --isolated` when embedding dependencies are available.
