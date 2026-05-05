# Troubleshooting

Run:

```bash
arn doctor
```

Common issues:

- **Missing Python 3.10+** — install a newer Python.
- **Missing venv/pip on Linux** — run `sudo apt install python3-venv python3-pip`.
- **Model download failed** — check internet, DNS, disk space, and RAM.
- **OpenClaw hook not found** — rerun `python install.py` or copy files from `openclaw/hooks/arn-memory`.
- **Semantic recall is weak** — run `arn selftest --strict --isolated`; consider `arn models switch --tier base --download` on stronger hardware.
