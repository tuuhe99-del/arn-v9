# Installation

## Linux / macOS

```bash
bash install.sh --tier nano
```

## Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -tier nano
```

## Universal

```bash
python install.py --tier nano
```

## Verify

```bash
arn doctor
arn selftest --strict --isolated
```

If the self-test fails, the embedding model is not installed or could not load.
