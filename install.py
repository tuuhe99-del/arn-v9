#!/usr/bin/env python3
"""Cross-platform one-command installer for ARN + OpenClaw memory hooks."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = Path(os.environ.get("ARN_INSTALL_LOG", ROOT / "install.log")).expanduser()


def log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def info(msg: str) -> None:
    print(f"[ARN install] {msg}")
    log(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")
    log(f"[WARN] {msg}")


def fail(msg: str, code: int = 1) -> None:
    print(f"\n[FAILED] {msg}", file=sys.stderr)
    print(f"Installer log: {LOG}", file=sys.stderr)
    print("Common fixes:", file=sys.stderr)
    print("  - Install Python 3.10+ from python.org or your package manager.", file=sys.stderr)
    print("  - Make sure internet/DNS works so embedding models can download.", file=sys.stderr)
    print("  - On Linux: sudo apt install python3-pip python3-venv", file=sys.stderr)
    print("  - On Windows: run PowerShell as normal user and use: py install.py --tier nano", file=sys.stderr)
    print("  - On a Raspberry Pi, start with: python install.py --tier nano", file=sys.stderr)
    raise SystemExit(code)


def run(cmd: list[str], *, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    info("Running: " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(proc.stdout)
    if check and proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        if tail:
            print(tail, file=sys.stderr)
        fail(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return proc


def venv_python(venv_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def script_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if platform.system().lower().startswith("win") else "bin")


def create_shims(py: Path) -> None:
    home = Path.home()
    if platform.system().lower().startswith("win"):
        bin_dir = home / ".arn" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "arn.cmd").write_text(f'@echo off\r\n"{py}" -m arn.scripts.arn_cli %*\r\n', encoding="utf-8")
        (bin_dir / "arn.ps1").write_text(f'& "{py}" -m arn.scripts.arn_cli @args\n', encoding="utf-8")
        # Auto-add to user PATH via registry (no admin rights needed)
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                                 winreg.KEY_READ | winreg.KEY_WRITE)
            current, _ = winreg.QueryValueEx(key, "PATH")
            bin_str = str(bin_dir)
            if bin_str not in current:
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, current + ";" + bin_str)
                info("Added ARN to your Windows PATH. Restart your terminal and type: arn check")
            winreg.CloseKey(key)
        except Exception as exc:
            warn(f"Could not auto-add to PATH ({exc}). Add manually: {bin_dir}")
    else:
        bin_dir = home / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        shim = bin_dir / "arn"
        shim.write_text(f'#!/usr/bin/env bash\nexec "{py}" -m arn.scripts.arn_cli "$@"\n', encoding="utf-8")
        shim.chmod(0o755)
        # Auto-add to shell profile if not already on PATH
        if str(bin_dir) not in os.environ.get("PATH", ""):
            line = f'\nexport PATH="{bin_dir}:$PATH"  # added by ARN installer\n'
            for profile in ["~/.bashrc", "~/.zshrc", "~/.profile"]:
                p = Path(profile).expanduser()
                if p.exists() and str(bin_dir) not in p.read_text(errors="ignore"):
                    with p.open("a") as f:
                        f.write(line)
                    info(f"Added 'arn' to PATH in {p}. Restart terminal or run: source {p}")
                    break


def install_openclaw_files() -> None:
    home = Path.home()
    skill_dir = Path(os.environ.get("OPENCLAW_SKILLS_DIR", home / ".openclaw" / "skills" / "arn-memory")).expanduser()
    hook_dir = Path(os.environ.get("OPENCLAW_HOOK_DIR", home / ".openclaw" / "hooks" / "arn-memory")).expanduser()
    skill_dir.mkdir(parents=True, exist_ok=True)
    hook_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "arn" / "openclaw_skill" / "SKILL.md", skill_dir / "SKILL.md")
    shutil.copy2(ROOT / "openclaw" / "hooks" / "arn-memory" / "HOOK.md", hook_dir / "HOOK.md")
    shutil.copy2(ROOT / "openclaw" / "hooks" / "arn-memory" / "handler.ts", hook_dir / "handler.ts")
    info(f"Installed OpenClaw skill: {skill_dir}")
    info(f"Installed OpenClaw hook:  {hook_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install ARN for Linux, macOS, or Windows.")
    parser.add_argument("--tier", default=os.environ.get("ARN_EMBEDDING_TIER", "nano"), choices=["nano", "small", "balanced", "base", "base-e5", "large"], help="Embedding model tier to install/download.")
    parser.add_argument("--venv", default=os.environ.get("ARN_VENV_DIR", str(ROOT / ".venv")), help="Virtual environment directory.")
    parser.add_argument("--system", action="store_true", help="Install into the active Python instead of a local .venv.")
    parser.add_argument("--skip-model-download", action="store_true", help="Install code/deps only; do not verify/download the embedding model.")
    parser.add_argument("--skip-openclaw", action="store_true", help="Do not copy OpenClaw skill/hook files.")
    args = parser.parse_args()

    LOG.write_text("", encoding="utf-8")
    info(f"Installing from {ROOT}")
    info(f"Detected OS: {platform.system()} {platform.release()} | Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        fail(f"Python {sys.version.split()[0]} found, but ARN requires Python 3.10+.")

    if args.system:
        py = Path(sys.executable)
        warn("Installing into the active Python environment because --system was used.")
    else:
        venv_dir = Path(args.venv).expanduser().resolve()
        info(f"Creating/updating virtual environment: {venv_dir}")
        try:
            venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(str(venv_dir))
        except Exception as exc:
            fail(f"Could not create virtual environment: {exc}")
        py = venv_python(venv_dir)
        if not py.exists():
            fail(f"Virtual environment Python was not found at {py}")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(py), "-m", "pip", "install", "-e", str(ROOT)])
    create_shims(py)

    env = os.environ.copy()
    env["ARN_EMBEDDING_TIER"] = args.tier

    if not args.skip_model_download:
        MAX_ATTEMPTS = 3
        for attempt in range(1, MAX_ATTEMPTS + 1):
            info(f"Downloading memory model '{args.tier}' (attempt {attempt}/{MAX_ATTEMPTS})...")
            proc = run([str(py), "-m", "arn.scripts.arn_cli", "models", "download",
                        "--tier", args.tier], env=env, check=False)
            if proc.returncode == 0:
                break
            if attempt < MAX_ATTEMPTS:
                info("Download failed — waiting 5 seconds and trying again...")
                import time; time.sleep(5)
        else:
            fail(
                f"Could not download the memory model after {MAX_ATTEMPTS} attempts.\n"
                "Common causes:\n"
                "  - No internet connection\n"
                "  - Not enough disk space (need ~100 MB for nano)\n"
                "  - Low RAM (try --tier nano on older machines)\n"
                "You can retry later with: arn models download --tier nano"
            )
    else:
        warn("Skipped model download. Run later: arn models download --tier nano")

    run([str(py), "-m", "arn.scripts.arn_cli", "models", "switch", "--tier", args.tier], env=env)

    if not args.skip_openclaw:
        install_openclaw_files()
        openclaw = shutil.which("openclaw")
        if openclaw:
            proc = run([openclaw, "hooks", "enable", "arn-memory"], check=False)
            if proc.returncode == 0:
                info("Enabled OpenClaw hook: arn-memory")
            else:
                warn("OpenClaw CLI was found, but hook auto-enable failed. Run: openclaw hooks list && openclaw hooks enable arn-memory")
        else:
            warn("OpenClaw CLI was not found. Hook files were copied; enable them when OpenClaw is installed.")

    # Final non-strict doctor so user gets helpful report even if OpenClaw is absent.
    run([str(py), "-m", "arn.scripts.arn_cli", "doctor"], env=env, check=False)

    print("\n✅ ARN install complete.\n")
    print("Next commands:")
    print("  arn models list")
    print("  arn check")
    print("  arn doctor")
    print("  arn store --content \"User prefers beginner-friendly setup\" --importance 0.8 --tags preference")
    print("\nIf OpenClaw is installed, restart your OpenClaw gateway after enabling the hook.")
    print(f"Installer log: {LOG}")


if __name__ == "__main__":
    main()
