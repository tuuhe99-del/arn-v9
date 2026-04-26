#!/usr/bin/env python3
"""
ARN CLI — main entrypoint for the `arn` command.

Usage:
    arn recall "what does the user prefer?"
    arn store "User prefers Python" --importance 0.8
    arn context "project architecture decisions"
    arn perceive "some fact" --source conversation
    arn stats
    arn connect          # interactive setup wizard
    arn ping             # check daemon status
    arn daemon start|stop|status
"""

import sys
import os
import argparse
import json
from pathlib import Path


# ── data dir resolution ────────────────────────────────────────────────────────
def _data_dir() -> Path:
    d = os.environ.get("ARN_DATA_DIR") or str(Path.home() / ".arn_data" / "default")
    return Path(d)


# ── model tier helpers ─────────────────────────────────────────────────────────
MODEL_TIERS = {
    "nano": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "22MB",
        "ram": "~90MB",
        "speed": "~30ms/encode",
        "quality": "good",
        "notes": "Raspberry Pi, low-power devices, anything with <1GB free RAM",
    },
    "small": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "size": "420MB",
        "ram": "~420MB",
        "speed": "~60ms/encode",
        "quality": "better",
        "notes": "Mid-range laptop/desktop, Pi 5 with 8GB RAM",
    },
    "base": {
        "model": "BAAI/bge-base-en-v1.5",
        "size": "440MB",
        "ram": "~500MB",
        "speed": "~80ms/encode",
        "quality": "best retrieval",
        "notes": "Desktop/server, recommended if you have RAM",
    },
    "large": {
        "model": "BAAI/bge-large-en-v1.5",
        "size": "1.3GB",
        "ram": "~1.3GB",
        "speed": "~120ms/encode",
        "quality": "high quality",
        "notes": "High-end desktop or server with 4GB+ free RAM",
    },
    "xl": {
        "model": "intfloat/e5-mistral-7b-instruct",
        "size": "~14GB",
        "ram": "~2.5GB fp16 / 14GB+ fp32",
        "speed": "~300ms+ CPU, ~50ms GPU",
        "quality": "top-tier (MTEB)",
        "notes": "GPU strongly recommended. Server/workstation only.",
    },
}


def _detect_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        pass
    return 4.0  # safe default


def _suggest_tier() -> str:
    ram = _detect_ram_gb()
    if ram < 2:
        return "nano"
    elif ram < 6:
        return "small"
    elif ram < 12:
        return "base"
    else:
        return "large"  # suggest large for machines with 12GB+ RAM


# ── connect wizard ─────────────────────────────────────────────────────────────
def cmd_connect(args):
    print("\n╔══════════════════════════════════════════╗")
    print("║     ARN — Setup Wizard                   ║")
    print("╚══════════════════════════════════════════╝\n")

    # 1. Data directory
    default_data = str(Path.home() / ".arn_data" / "default")
    current = os.environ.get("ARN_DATA_DIR", default_data)
    print(f"Memory data directory [{current}]: ", end="")
    user_data = input().strip() or current
    data_dir = Path(user_data)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {data_dir}\n")

    # 2. Model tier
    ram = _detect_ram_gb()
    suggested = _suggest_tier()
    print(f"Detected RAM: {ram:.1f} GB → suggested tier: {suggested}\n")
    print("Choose embedding model tier:")
    for tier, info in MODEL_TIERS.items():
        marker = " ← suggested" if tier == suggested else ""
        print(f"  [{tier}]  {info['model']}")
        print(f"         Size: {info['size']} | RAM: {info['ram']} | {info['notes']}{marker}")
    print()
    print(f"Tier (nano/small/base) [{suggested}]: ", end="")
    chosen = input().strip().lower() or suggested
    if chosen not in MODEL_TIERS:
        chosen = suggested
    model_name = MODEL_TIERS[chosen]["model"]
    print(f"  ✓ {model_name}\n")

    # 3. Framework detection
    framework = _detect_framework()
    print(f"Detected framework: {framework or 'none detected'}")
    if not framework:
        print("Framework (openclaw/langchain/raw) [raw]: ", end="")
        framework = input().strip().lower() or "raw"
    print()

    # 4. Shell RC update
    shell_rc = Path.home() / (".zshrc" if "zsh" in os.environ.get("SHELL", "") else ".bashrc")
    lines = [
        f'export ARN_DATA_DIR="{data_dir}"',
        f'export ARN_EMBEDDING_MODEL="{model_name}"',
        f'alias arn="python3 -m arn.cli"',
    ]
    existing = shell_rc.read_text() if shell_rc.exists() else ""
    added = []
    with open(shell_rc, "a") as f:
        for line in lines:
            if line not in existing:
                f.write(line + "\n")
                added.append(line)
    if added:
        print(f"  Added to {shell_rc}:")
        for l in added:
            print(f"    {l}")
    else:
        print(f"  {shell_rc} already configured.")
    print()

    # 5. Framework wiring
    _wire_framework(framework, data_dir, model_name)

    # 6. Verify
    print("Running quick verify...")
    _quick_verify(data_dir, model_name)

    print("\n╔══════════════════════════════════════════╗")
    print("║  ARN ready. Restart your shell then:    ║")
    print("║  arn store \"hello\" --importance 0.5     ║")
    print("║  arn recall \"hello\"                     ║")
    print("╚══════════════════════════════════════════╝\n")


def _detect_framework() -> str:
    ocp = Path.home() / ".openclaw"
    if ocp.exists():
        return "openclaw"
    try:
        import langchain  # noqa
        return "langchain"
    except ImportError:
        pass
    return ""


def _wire_framework(framework: str, data_dir: Path, model_name: str):
    if framework == "openclaw":
        _wire_openclaw(data_dir, model_name)
    elif framework == "langchain":
        _wire_langchain(data_dir, model_name)
    else:
        _wire_raw(data_dir, model_name)


def _wire_openclaw(data_dir: Path, model_name: str):
    skills_base = Path.home() / ".openclaw" / "workspace"
    # find agent dirs
    agents_dir = skills_base / "agents"
    skill_src = Path(__file__).parent.parent / "openclaw_skill" / "SKILL.md"
    if not skill_src.exists():
        # try alternate location
        skill_src = Path(__file__).parent / "core" / "openclaw_skill" / "SKILL.md"

    count = 0
    if agents_dir.exists() and skill_src.exists():
        for agent_dir in agents_dir.iterdir():
            if agent_dir.is_dir():
                dest = agent_dir / "skills" / "arn-memory"
                dest.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(skill_src, dest / "SKILL.md")
                count += 1
    if count:
        print(f"  ✓ OpenClaw: ARN memory skill deployed to {count} agents")
    else:
        print("  ✓ OpenClaw detected — copy arn/openclaw_skill/SKILL.md to your agent skills dirs manually")


def _wire_langchain(data_dir: Path, model_name: str):
    example = f"""
# Add to your LangChain agent setup:
from arn import ARNv9
from langchain.tools import tool

arn = ARNv9(data_dir="{data_dir}")

@tool
def remember(fact: str) -> str:
    "Store a fact worth remembering about the user."
    arn.perceive(fact, importance=0.7)
    return "stored"

@tool
def recall(query: str) -> str:
    "Recall relevant context before answering."
    hits = arn.recall(query, top_k=3)
    return "\\n".join(h["content"] for h in hits) or "nothing recalled"
"""
    snippet_path = Path.home() / "arn_langchain_snippet.py"
    snippet_path.write_text(example)
    print(f"  ✓ LangChain: example snippet written to {snippet_path}")


def _wire_raw(data_dir: Path, model_name: str):
    system_prompt_addition = f"""
You have persistent memory via ARN.

Store a fact:
  python3 -m arn.cli store "<fact>" --importance 0.8

Recall by meaning:
  python3 -m arn.cli recall "<question>"

Get context for a topic:
  python3 -m arn.cli context "<topic>"

Store when: user shares preferences, makes decisions, or says something worth remembering.
Recall before answering anything where past context matters.
Memory dir: {data_dir}
"""
    prompt_path = Path.home() / "arn_system_prompt_addition.txt"
    prompt_path.write_text(system_prompt_addition)
    print(f"  ✓ Raw setup: add contents of {prompt_path} to your system prompt")


def _quick_verify(data_dir: Path, model_name: str):
    try:
        os.environ["ARN_DATA_DIR"] = str(data_dir)
        os.environ["ARN_EMBEDDING_MODEL"] = model_name
        from arn.core import ARNv9
        arn = ARNv9(data_dir=str(data_dir))
        arn.perceive("ARN setup verification fact", importance=0.1)
        hits = arn.recall("setup verification", top_k=1)
        arn.close()
        if hits:
            print("  ✓ Store + recall working")
        else:
            print("  ⚠ Store worked but recall returned nothing — first run may need model download")
    except Exception as e:
        print(f"  ⚠ Verify failed: {e}")
        print("    This may be fine — model downloads on first real use")


# ── core commands ──────────────────────────────────────────────────────────────
def _get_plugin():
    """Return an ARNPlugin instance (supports temporal tagging + full API)."""
    data_dir = str(_data_dir())
    # ARN_EMBEDDING_TIER is the canonical var; fall back to translating ARN_EMBEDDING_MODEL
    tier = os.environ.get("ARN_EMBEDDING_TIER", None)
    if tier is None:
        model = os.environ.get("ARN_EMBEDDING_MODEL", "")
        if "mpnet" in model:
            tier = "small"
        elif "bge-base" in model:
            tier = "base"
        elif "e5-base" in model:
            tier = "base-e5"
        else:
            tier = "nano"
    from arn.plugin import ARNPlugin
    kwargs = {"agent_id": "cli", "data_root": str(Path(data_dir).parent)}
    if tier:
        kwargs["embedding_tier"] = tier
    return ARNPlugin(**kwargs)


def _get_arn():
    """Return a raw ARNv9 instance (for recall/context/stats)."""
    data_dir = str(_data_dir())
    model = os.environ.get("ARN_EMBEDDING_MODEL", "")
    from arn.core import ARNv9
    kwargs = {"data_dir": data_dir}
    if model:
        kwargs["embedding_model"] = model
    return ARNv9(**kwargs)


def cmd_store(args):
    p = _get_plugin()
    p.store(
        args.content,
        importance=args.importance,
        source=args.source,
        time_context=args.time_context,
    )
    p.shutdown()
    print(f"stored (importance={args.importance}, time_context={args.time_context})")


def cmd_recall(args):
    p = _get_plugin()
    hits = p.recall(args.query, top_k=args.top_k)
    p.shutdown()
    if not hits:
        print("(nothing recalled)")
        return
    for h in hits:
        score = h.get("score", 0)
        content = h.get("content", "")
        tc = h.get("time_context", "")
        tc_tag = f" [{tc}]" if tc and tc != "current" else ""
        print(f"[{score:.2f}]{tc_tag} {content}")


def cmd_context(args):
    p = _get_plugin()
    ctx = p.get_context_window(args.query, max_tokens=args.max_tokens)
    p.shutdown()
    print(ctx or "(no context)")


def cmd_perceive(args):
    """Alias for store — passive absorption mode."""
    p = _get_plugin()
    p.store(args.content, importance=args.importance, source=args.source)
    p.shutdown()
    print(f"absorbed (importance={args.importance})")


def cmd_stats(args):
    p = _get_plugin()
    stats = p.get_stats() if hasattr(p, "get_stats") else {}
    p.shutdown()
    if stats:
        print(json.dumps(stats, indent=2))
    else:
        print("Stats not available")


def cmd_ping(args):
    sock = "/tmp/arn_daemon.sock"
    import socket, time
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(sock)
        s.sendall(json.dumps({"cmd": "ping"}).encode() + b"\n")
        resp = json.loads(s.recv(4096).decode())
        s.close()
        if resp.get("status") == "ok":
            print("daemon: running ✓")
        else:
            print(f"daemon: unexpected response: {resp}")
    except Exception:
        print("daemon: not running (use: arn daemon start)")


def cmd_daemon(args):
    daemon_script = Path(__file__).parent.parent.parent / "arn_daemon.py"
    if not daemon_script.exists():
        # try installed location
        import shutil
        daemon_script = shutil.which("arn-daemon") or ""
    if args.action == "start":
        import subprocess
        subprocess.Popen(
            [sys.executable, str(daemon_script), "start"],
            stdout=open(Path.home() / ".arn_daemon.log", "w"),
            stderr=subprocess.STDOUT,
        )
        print("daemon starting... (check ~/.arn_daemon.log)")
    elif args.action == "stop":
        import subprocess
        subprocess.run([sys.executable, str(daemon_script), "stop"])
    elif args.action == "status":
        cmd_ping(args)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="arn",
        description="ARN — brain-inspired persistent memory for AI agents",
    )
    sub = parser.add_subparsers(dest="command")

    # connect
    p_connect = sub.add_parser("connect", help="Interactive setup wizard")
    p_connect.set_defaults(func=cmd_connect)

    # store
    p_store = sub.add_parser("store", help="Store a fact")
    p_store.add_argument("content", help="The fact to store")
    p_store.add_argument("--importance", type=float, default=0.5)
    p_store.add_argument("--source", default="user")
    p_store.add_argument("--time-context", default="current", dest="time_context")
    p_store.set_defaults(func=cmd_store)

    # recall
    p_recall = sub.add_parser("recall", help="Recall by meaning")
    p_recall.add_argument("query", help="What to recall")
    p_recall.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_recall.set_defaults(func=cmd_recall)

    # context
    p_ctx = sub.add_parser("context", help="Get context block for a topic")
    p_ctx.add_argument("query")
    p_ctx.add_argument("--max-tokens", type=int, default=1200, dest="max_tokens")
    p_ctx.set_defaults(func=cmd_context)

    # perceive
    p_perceive = sub.add_parser("perceive", help="Passively absorb a fact")
    p_perceive.add_argument("content")
    p_perceive.add_argument("--importance", type=float, default=0.3)
    p_perceive.add_argument("--source", default="conversation")
    p_perceive.set_defaults(func=cmd_perceive)

    # stats
    p_stats = sub.add_parser("stats", help="Show memory stats")
    p_stats.set_defaults(func=cmd_stats)

    # ping
    p_ping = sub.add_parser("ping", help="Check daemon status")
    p_ping.set_defaults(func=cmd_ping)

    # daemon
    p_daemon = sub.add_parser("daemon", help="Control the ARN daemon")
    p_daemon.add_argument("action", choices=["start", "stop", "status"])
    p_daemon.set_defaults(func=cmd_daemon)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
