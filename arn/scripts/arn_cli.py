#!/usr/bin/env python3
"""ARN CLI — local semantic memory for AI agents."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _package_root)

from arn.config import CONFIG_PATH, get_default_tier, set_default_tier
from arn.core.embeddings import (
    MODEL_CONFIGS,
    download_model,
    get_total_ram_mb,
    model_table,
    normalize_tier,
    recommend_tier,
)
from arn.plugin import ARNPlugin
from arn.storage.persistence import StorageEngine

DEFAULT_DATA_ROOT = os.environ.get("ARN_DATA_ROOT", os.path.expanduser("~/.arn_data"))
DEFAULT_AGENT_ID  = os.environ.get("OPENCLAW_AGENT_ID", "default")
DEFAULT_TIER      = get_default_tier("nano")


def emit_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def get_plugin(args: argparse.Namespace) -> ARNPlugin:
    tier = normalize_tier(getattr(args, "embedding_tier", None) or DEFAULT_TIER)
    plugin = ARNPlugin(
        agent_id=getattr(args, "agent_id", None) or DEFAULT_AGENT_ID,
        data_root=getattr(args, "data_root", None) or DEFAULT_DATA_ROOT,
        embedding_tier=tier,
        auto_consolidate=True,
        consolidation_threshold=128,
    )
    if plugin._arn.embedder.is_degraded:
        msg = (
            "WARNING: memory model not loaded — semantic recall won't work.\n"
            "  Fix: arn models download --tier nano"
        )
        if getattr(args, "strict", False):
            print(msg, file=sys.stderr)
            plugin.shutdown()
            raise SystemExit(1)
        print(msg, file=sys.stderr)
    return plugin


def parse_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


def _import_status(module_name: str) -> dict:
    try:
        module = __import__(module_name)
        return {"ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ─── Privacy / memory management ──────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        memories = plugin.list_memories(limit=args.limit, memory_type=args.type)
    if not memories:
        print("No memories stored yet.")
        return
    import datetime
    for m in memories:
        ts = datetime.datetime.fromtimestamp(m["created_at"]).strftime("%Y-%m-%d %H:%M")
        badge = "📌" if m["type"] == "semantic" else "💭"
        imp = f"  imp={m['importance']:.1f}" if "importance" in m else f"  conf={m.get('confidence', 0):.2f}"
        print(f"{badge} [{m['id']}] {ts}{imp}")
        print(f"   {m['content'][:120]}{'...' if len(m['content']) > 120 else ''}")


def cmd_search(args: argparse.Namespace) -> None:
    query = _arg_text(args, "query", "query_pos")
    with get_plugin(args) as plugin:
        results = plugin.search_memories(query, limit=args.limit)
    if not results:
        print(f"No memories found containing: {query}")
        return
    import datetime
    print(f"Found {len(results)} match{'es' if len(results) != 1 else ''}:")
    for m in results:
        ts = datetime.datetime.fromtimestamp(m["created_at"]).strftime("%Y-%m-%d %H:%M")
        print(f"  [{m['id']}] {ts}  {m['content'][:120]}{'...' if len(m['content']) > 120 else ''}")


def cmd_forget(args: argparse.Namespace) -> None:
    if args.all:
        confirm = input("This will delete ALL memories for this agent. Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("Cancelled.")
            return
    with get_plugin(args) as plugin:
        result = plugin.forget(
            episode_ids=[int(i) for i in args.ids] if args.ids else None,
            forget_all=args.all,
        )
    n = result["deleted_episodes"]
    print(f"Deleted {n} episodic memor{'y' if n == 1 else 'ies'}.")


# ─── Core memory commands ──────────────────────────────────────────────────────

def _arg_text(args: argparse.Namespace, primary: str, fallback: str) -> str:
    value = getattr(args, primary, None) or getattr(args, fallback, None) or ""
    value = value.strip() if isinstance(value, str) else value
    if not value:
        raise SystemExit(f"Missing text. Use --{primary.replace('_', '-')} or pass it as a quoted argument.")
    return value


def cmd_graph(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        if args.graph_cmd == "query":
            results = plugin.graph_query(args.entity,
                                          include_superseded=args.include_superseded)
            if not results:
                print(f"No facts known about '{args.entity}'")
                return
            print(f"Facts about '{args.entity}':")
            for f in results:
                status = " [superseded]" if f.get("superseded") else ""
                print(f"  {f['subject']} --[{f['relation']}]--> {f['object']}{status}")
        elif args.graph_cmd == "history":
            results = plugin.fact_history(args.entity, args.relation)
            if not results:
                print(f"No history for {args.entity}/{args.relation}")
                return
            import datetime
            print(f"History of '{args.entity}' → '{args.relation}':")
            for f in results:
                ts = datetime.datetime.fromtimestamp(f["created_at"]).strftime("%Y-%m-%d %H:%M")
                status = " ← current" if not f.get("superseded") else "   [old]"
                print(f"  {ts}  {f['object']}{status}")
        elif args.graph_cmd == "summary":
            emit_json(plugin.graph_summary())


def cmd_store(args: argparse.Namespace) -> None:
    content = _arg_text(args, "content", "content_pos")
    with get_plugin(args) as plugin:
        emit_json(plugin.store(
            content=content,
            importance=args.importance,
            tags=parse_tags(args.tags),
            source=args.source,
            time_context=args.time_context,
            memory_type=args.memory_type,
            scope=args.scope,
            priority=args.priority,
            name=args.name,
        ))


def cmd_recall(args: argparse.Namespace) -> None:
    query = _arg_text(args, "query", "query_pos")
    with get_plugin(args) as plugin:
        emit_json(plugin.recall(query=query, top_k=args.top_k,
                                time_filter=args.time_filter))


def cmd_context(args: argparse.Namespace) -> None:
    query = getattr(args, "query", None) or getattr(args, "query_pos", None) or ""
    with get_plugin(args) as plugin:
        print(plugin.build_context_packet(query=query or None,
                                          agent=args.for_agent or plugin.agent_id,
                                          task=args.task,
                                          max_tokens=args.max_tokens))


def cmd_maintain(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        emit_json(plugin.maintain())


def cmd_stats(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        emit_json(plugin.get_stats())



# ─── Human memory commands ────────────────────────────────────────────────────

def _print_memories(rows: list[dict], empty: str = "No matching memories.") -> None:
    if not rows:
        print(empty)
        return
    import datetime
    for r in rows:
        ts = datetime.datetime.fromtimestamp(r.get("created_at", 0)).strftime("%Y-%m-%d %H:%M")
        priority = r.get("priority", (r.get("context") or {}).get("priority", "normal"))
        scope = r.get("scope", (r.get("context") or {}).get("scope", "global"))
        print(f"[{r.get('id','?')}] {ts} scope={scope} priority={priority}")
        print(r.get("content", "").rstrip())
        print()


def cmd_identity(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        if args.identity_action == "set":
            emit_json(plugin.set_identity(args.agent, name=args.name, role=args.role,
                                          must=args.must, must_not=args.must_not))
        elif args.identity_action == "show":
            _print_memories(plugin.get_identity(args.agent), f"No identity set for {args.agent}.")


def cmd_rule(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        if args.rule_action == "add":
            emit_json(plugin.add_rule(args.agent, args.rule, priority=args.priority))
        elif args.rule_action == "list":
            _print_memories(plugin.list_rules(args.agent), "No rules found.")


def cmd_procedure(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        if args.procedure_action == "add":
            emit_json(plugin.add_procedure(args.name, args.step, success=args.success,
                                           agent=args.agent))
        elif args.procedure_action == "recall":
            rows = plugin.recall_procedures(args.query, agent=args.agent, top_k=args.top_k)
            _print_memories(rows, "No procedures found.")
        elif args.procedure_action == "list":
            rows = plugin._recent_by_type("procedure",
                                          scopes=["global"] + ([f"agent:{args.agent}"] if args.agent else []),
                                          limit=args.limit)
            _print_memories(rows, "No procedures stored.")


def cmd_error(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        if args.error_action == "add":
            emit_json(plugin.add_error_lesson(mistake=args.mistake, fix=args.fix,
                                             lesson=args.lesson, agent=args.agent,
                                             task=args.task))
        elif args.error_action == "list":
            _print_memories(plugin.list_error_lessons(agent=args.agent, query=args.query,
                                                      limit=args.limit),
                            "No error/lesson memories found.")

def cmd_share(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        if args.share_action == "send":
            content = _arg_text(args, "content", "content_pos")
            emit_json(plugin.share_memory(
                content=content,
                to_agents=args.to,
                from_agent=args.from_agent or plugin.agent_id,
                task=args.task,
                importance=args.importance,
                priority=args.priority,
                tags=parse_tags(args.tags),
            ))
        elif args.share_action in ("inbox", "outbox", "list"):
            direction = None
            if args.share_action == "inbox":
                direction = "inbox"
            elif args.share_action == "outbox":
                direction = "outbox"
            rows = plugin.list_shared_memories(
                agent=args.agent or plugin.agent_id,
                direction=direction,
                query=args.query,
                task=args.task,
                limit=args.limit,
            )
            _print_memories(rows, "No shared memories found.")

# ─── Diagnostics ──────────────────────────────────────────────────────────────

def cmd_check(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        stats = plugin.get_stats()
        ok = not plugin._arn.embedder.is_degraded
        emit_json({
            "ok": ok,
            "agent_id": plugin.agent_id,
            "embedding_tier": plugin._arn.embedder.tier,
            "embedding_model": plugin._arn.embedder.model_name,
            "episodic_count": stats["episodic_count"],
            "semantic_count": stats["semantic_count"],
            "storage_mb": round(stats["storage"]["total_size_mb"], 2),
            "message": "Ready" if ok else "Model not loaded. Run: arn models download --tier nano",
        })


def cmd_doctor(args: argparse.Namespace) -> None:
    report: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "ok": sys.version_info >= (3, 10),
        },
        "platform": {
            "system": sys.platform,
            "total_ram_mb": get_total_ram_mb(),
            "recommended_tier": recommend_tier(),
        },
        "commands": {
            "arn": shutil.which("arn"),
            "openclaw": shutil.which("openclaw"),
        },
        "imports": {
            "numpy": _import_status("numpy"),
            "sentence_transformers": _import_status("sentence_transformers"),
            "torch": _import_status("torch"),
        },
        "env": {
            "ARN_DATA_ROOT": getattr(args, "data_root", None) or DEFAULT_DATA_ROOT,
            "ARN_EMBEDDING_TIER": getattr(args, "embedding_tier", None) or DEFAULT_TIER,
        },
        "openclaw_paths": {
            "skill": os.path.expanduser("~/.openclaw/skills/arn-memory/SKILL.md"),
            "hook": os.path.expanduser("~/.openclaw/hooks/arn-memory/HOOK.md"),
        },
    }
    try:
        with get_plugin(args) as plugin:
            report["arn"] = {
                "ok": not plugin._arn.embedder.is_degraded,
                "agent_id": plugin.agent_id,
                "embedding_tier": plugin._arn.embedder.tier,
                "embedding_model": plugin._arn.embedder.model_name,
                "embedding_dim": plugin._arn.embedder.embedding_dim,
            }
    except SystemExit:
        report["arn"] = {"ok": False, "error": "embeddings unavailable"}
    except Exception as exc:
        report["arn"] = {"ok": False, "error": str(exc)}

    problems, fixes = [], []
    if not report["python"]["ok"]:
        problems.append("Python is too old (need 3.10 or newer).")
        fixes.append("Download the latest Python from python.org and reinstall.")
    for dep, label, fix in [
        ("numpy", "numpy", "pip install numpy"),
        ("sentence_transformers", "sentence-transformers", "pip install sentence-transformers"),
        ("torch", "torch (PyTorch)", "pip install torch"),
    ]:
        if not report["imports"][dep]["ok"]:
            problems.append(f"Missing package: {label}.")
            fixes.append(f"Run: {fix}")
    if not report.get("arn", {}).get("ok"):
        problems.append("The memory model didn't load.")
        fixes.append("Run: python install.py --tier nano")
    if not os.path.exists(report["openclaw_paths"]["hook"]):
        problems.append("OpenClaw hook not installed.")
        fixes.append("Run: python install.py")
    if not os.path.exists(report["openclaw_paths"]["skill"]):
        problems.append("OpenClaw skill not installed.")
        fixes.append("Run: python install.py")

    report["ok"] = len(problems) == 0
    report["problems"] = problems

    if report["ok"]:
        print("✓ Everything looks good. ARN is ready.")
    else:
        print(f"Found {len(problems)} problem(s):\n")
        for i, (p, f) in enumerate(zip(problems, fixes), 1):
            print(f"  {i}. {p}")
            print(f"     Fix: {f}")
        print()

    if getattr(args, "json", False):
        emit_json(report)
    if getattr(args, "strict", False) and problems:
        raise SystemExit(1)


def cmd_selftest(args: argparse.Namespace) -> None:
    import tempfile
    if getattr(args, "isolated", False):
        tmp = tempfile.mkdtemp(prefix="arn_selftest_")
        plugin = ARNPlugin(agent_id="selftest", data_root=tmp,
                           embedding_tier=DEFAULT_TIER, auto_consolidate=False)
    else:
        plugin = get_plugin(args)
    try:
        result = plugin.semantic_selftest()
        emit_json(result)
        if not result["ok"]:
            raise SystemExit(1)
    finally:
        plugin.shutdown()
        if getattr(args, "isolated", False):
            import shutil as _shutil
            _shutil.rmtree(tmp, ignore_errors=True)


def cmd_contradictions(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        emit_json(plugin.list_contradictions(limit=args.limit))


# ─── Update check ─────────────────────────────────────────────────────────────

def cmd_update_check(args: argparse.Namespace) -> None:
    """Offline-safe update check.

    ARN does not phone home by default. If a project maintainer wants update
    checks, set ARN_UPDATE_URL to a raw VERSION file URL.
    """
    from arn import __version__ as current
    update_url = os.environ.get("ARN_UPDATE_URL")
    print(f"Installed: {current}")
    if not update_url:
        print("Update URL not configured. Set ARN_UPDATE_URL to enable update checks.")
        return
    try:
        import urllib.request
        with urllib.request.urlopen(update_url, timeout=5) as r:
            latest = r.read().decode().strip()
        if latest == current:
            print(f"Latest:    {latest}")
            print("✓ You are up to date.")
        else:
            print(f"Latest:    {latest}  ← update available")
            print("To update: download the latest release and reinstall.")
    except Exception as exc:
        print(f"Could not check for updates: {exc}")


# ─── Version ──────────────────────────────────────────────────────────────────

def cmd_version(args: argparse.Namespace) -> None:
    from arn import __version__
    print(f"ARN {__version__}")


# ─── Models ───────────────────────────────────────────────────────────────────

def _migrate_model_tier(args: argparse.Namespace) -> None:
    from_tier = normalize_tier(args.from_tier)
    to_tier   = normalize_tier(args.to_tier)
    if from_tier == to_tier:
        raise SystemExit("from-tier and to-tier are the same; nothing to migrate.")

    if getattr(args, "download", False):
        result = download_model(to_tier, cache_folder=getattr(args, "cache_folder", None))
        if not result["ok"]:
            raise SystemExit(f"Could not download model for tier '{to_tier}': {result['error']}")

    data_root = getattr(args, "data_root", None) or DEFAULT_DATA_ROOT
    agent_id  = getattr(args, "agent_id", None) or DEFAULT_AGENT_ID

    from_plugin = ARNPlugin(agent_id=agent_id, data_root=data_root,
                            embedding_tier=from_tier, auto_consolidate=False)
    to_plugin   = ARNPlugin(agent_id=agent_id, data_root=data_root,
                            embedding_tier=to_tier, auto_consolidate=False)

    episodes = from_plugin._arn.storage.get_all_episodes(consolidated=None,
                                                          limit=args.limit)
    migrated = 0
    for ep in episodes:
        to_plugin.store(
            content=ep["content"],
            importance=ep["importance"],
            source=ep.get("source", "migration"),
            context=ep.get("context", {}),
            time_context=ep.get("context", {}).get("time_context", "current"),
        )
        migrated += 1

    from_plugin.shutdown()
    if getattr(args, "consolidate", False):
        to_plugin.maintain()
    to_plugin.shutdown()
    set_default_tier(to_tier)
    print(f"Migrated {migrated} episodes from '{from_tier}' to '{to_tier}'.")
    print(f"Default tier set to '{to_tier}'.")


def cmd_models(args: argparse.Namespace) -> None:
    action = args.model_action

    if action == "list":
        rows = model_table()
        if getattr(args, "json", False):
            emit_json(rows)
            return
        print(f"{'TIER':<10} {'DIM':>4} {'QUALITY':<12} {'SPEED':<8} "
              f"{'RAM MB':>7} {'DISK MB':>8} {'REC':>4}")
        print("-" * 62)
        for r in rows:
            rec = "◀" if r["recommended_here"] else ""
            print(f"{r['tier']:<10} {r['dim']:>4} {r['quality']:<12} {r['speed']:<8} "
                  f"{r['approx_ram_mb']:>7} {r['approx_disk_mb']:>8} {rec:>4}")

    elif action == "recommend":
        ram = get_total_ram_mb()
        tier = recommend_tier(ram)
        print(f"Detected RAM: {ram} MB" if ram else "Could not detect RAM.")
        print(f"Recommended tier: {tier}")
        print(f"Switch with: arn models switch --tier {tier} --download")

    elif action == "download":
        tier = normalize_tier(getattr(args, "tier", None) or DEFAULT_TIER)
        print(f"Downloading model for tier '{tier}'...")
        result = download_model(tier, cache_folder=getattr(args, "cache_folder", None))
        emit_json(result)
        if not result["ok"]:
            raise SystemExit(1)

    elif action == "switch":
        tier = normalize_tier(args.tier)
        if getattr(args, "download", False):
            result = download_model(tier,
                                    cache_folder=getattr(args, "cache_folder", None))
            if not result["ok"]:
                raise SystemExit(f"Download failed: {result['error']}")
        set_default_tier(tier)
        print(f"Default tier set to '{tier}'.")
        print(f"Restart ARN for the change to take effect.")

    elif action == "migrate":
        _migrate_model_tier(args)


# ─── Hook ─────────────────────────────────────────────────────────────────────

def cmd_hook(args: argparse.Namespace) -> None:
    with get_plugin(args) as plugin:
        event = args.event
        if event == "receive":
            emit_json(plugin.on_message_received(
                message=args.message, role=args.role,
                importance=args.importance, time_context=args.time_context))
        elif event == "before-reply":
            print(plugin.before_reply(
                query=args.query or args.message, max_tokens=args.max_tokens))
        elif event == "send":
            emit_json(plugin.on_message_sent(
                message=args.message, role=args.role,
                importance=args.importance, time_context=args.time_context))
        elif event == "tool-result":
            emit_json(plugin.on_tool_result(
                tool_name=args.tool_name, result=args.message,
                importance=args.importance, time_context=args.time_context))
        elif event == "preprocessed":
            if args.message:
                plugin.on_message_received(
                    message=args.message, role=args.role,
                    importance=args.importance, time_context=args.time_context)
            memory = plugin.before_reply(
                query=args.query or args.message, max_tokens=args.max_tokens)
            print(
                f"{memory}\n\n---\n\n{args.message}"
                if memory.strip() and args.message
                else (memory or args.message)
            )
        else:
            raise SystemExit(f"Unknown hook event: {event}")


# ─── Global args ──────────────────────────────────────────────────────────────

def add_global_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--embedding-tier", default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--strict", action="store_true")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arn",
        description="ARN — local semantic memory for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arn store "Mohamed prefers Python" --importance 0.8
  arn recall "what does the user like to code in?"
  arn list
  arn search --query "Python"
  arn forget --ids 3 7
  arn forget --all
  arn doctor
  arn update-check
  arn models list
  arn models switch --tier base --download
  arn share send "Build passes after CLI fix" --from-agent developer --to manager --task "ARN release"
  arn share inbox --agent manager --task "ARN release"
""",
    )
    add_global_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    # Memory commands
    p = sub.add_parser("store", help="Store a memory")
    p.add_argument("content_pos", nargs="?", help="Memory text, e.g. 'Mohamed prefers Python'")
    p.add_argument("--content", "-c", default=None)
    p.add_argument("--importance", "-i", type=float, default=0.5)
    p.add_argument("--tags", "-t", default="")
    p.add_argument("--source", "-s", default="agent")
    p.add_argument("--type", dest="memory_type", default="episode",
                   choices=["episode", "fact", "preference", "identity", "rule", "procedure",
                            "error", "lesson", "task", "decision", "conflict", "shared"])
    p.add_argument("--scope", default="global", help="global, user, project, or agent:<name>")
    p.add_argument("--priority", default="normal", choices=["low", "normal", "high", "critical"])
    p.add_argument("--name", default=None, help="Optional stable name for this memory")
    p.add_argument("--time-context", default="current",
                   choices=["past", "current", "future", "timeless"])
    p.set_defaults(func=cmd_store)

    p = sub.add_parser("recall", help="Recall relevant memories")
    p.add_argument("query_pos", nargs="?", help="Question/query to recall by meaning")
    p.add_argument("--query", "-q", default=None)
    p.add_argument("--top-k", "-k", type=int, default=5)
    p.add_argument("--time-filter",
                   choices=["past", "current", "future", "timeless"])
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("context", help="Get formatted memory block for prompt injection")
    p.add_argument("query_pos", nargs="?", help="Question/query for context injection")
    p.add_argument("--query", "-q", default="")
    p.add_argument("--max-tokens", "-m", type=int, default=1000)
    p.add_argument("--for-agent", default=None, help="Build context for this agent identity")
    p.add_argument("--task", default=None, help="Current task goal")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("list", help="List stored memories")
    p.add_argument("--limit", "-n", type=int, default=50)
    p.add_argument("--type", choices=["all", "episodic", "semantic"], default="all")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("search", help="Keyword search across memories")
    p.add_argument("query_pos", nargs="?", help="Keyword text")
    p.add_argument("--query", "-q", default=None)
    p.add_argument("--limit", "-n", type=int, default=20)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("forget", help="Delete memories")
    p.add_argument("--ids", nargs="+", metavar="ID",
                   help="Memory IDs to delete (from 'arn list')")
    p.add_argument("--all", action="store_true", help="Delete ALL memories")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("maintain", help="Run memory consolidation")
    p.set_defaults(func=cmd_maintain)

    # Diagnostics
    p = sub.add_parser("stats", help="Show memory statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("check", help="Quick health check")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("doctor", help="Diagnose and fix problems")
    p.add_argument("--json", action="store_true", help="Output full JSON report")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("selftest", help="Verify semantic recall works end-to-end")
    p.add_argument("--isolated", action="store_true")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("contradictions", help="List detected memory contradictions")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_contradictions)

    p = sub.add_parser("graph", help="Query the fact graph")
    graph_sub = p.add_subparsers(dest="graph_cmd", required=True)

    pg = graph_sub.add_parser("query", help="All facts about an entity")
    pg.add_argument("entity", help="Entity name, e.g. 'user'")
    pg.add_argument("--include-superseded", action="store_true")
    pg.set_defaults(func=cmd_graph)

    ph = graph_sub.add_parser("history", help="How a fact evolved over time")
    ph.add_argument("entity", help="Subject entity")
    ph.add_argument("relation", help="Relation type, e.g. 'preference'")
    ph.set_defaults(func=cmd_graph)

    ps = graph_sub.add_parser("summary", help="Graph size and entities")
    ps.set_defaults(func=cmd_graph)


    # Human memory commands
    p = sub.add_parser("identity", help="Set/show agent identity memory")
    id_sub = p.add_subparsers(dest="identity_action", required=True)
    ps = id_sub.add_parser("set", help="Set an agent identity")
    ps.add_argument("agent")
    ps.add_argument("--name", default=None)
    ps.add_argument("--role", default=None)
    ps.add_argument("--must", action="append", default=[])
    ps.add_argument("--must-not", action="append", default=[])
    ps.set_defaults(func=cmd_identity)
    ph = id_sub.add_parser("show", help="Show an agent identity")
    ph.add_argument("agent")
    ph.set_defaults(func=cmd_identity)

    p = sub.add_parser("rule", help="Manage rule memory")
    rule_sub = p.add_subparsers(dest="rule_action", required=True)
    pa = rule_sub.add_parser("add", help="Add a rule")
    pa.add_argument("agent", help="agent name, or 'global'")
    pa.add_argument("rule")
    pa.add_argument("--priority", default="high", choices=["low", "normal", "high", "critical"])
    pa.set_defaults(func=cmd_rule)
    pl = rule_sub.add_parser("list", help="List rules")
    pl.add_argument("agent", nargs="?", default=None)
    pl.set_defaults(func=cmd_rule)

    p = sub.add_parser("procedure", help="Manage procedural memory")
    proc_sub = p.add_subparsers(dest="procedure_action", required=True)
    pa = proc_sub.add_parser("add", help="Add a how-to procedure")
    pa.add_argument("name")
    pa.add_argument("--step", action="append", required=True)
    pa.add_argument("--success", default=None)
    pa.add_argument("--agent", default=None)
    pa.set_defaults(func=cmd_procedure)
    pr = proc_sub.add_parser("recall", help="Recall a procedure by meaning")
    pr.add_argument("query")
    pr.add_argument("--agent", default=None)
    pr.add_argument("--top-k", type=int, default=5)
    pr.set_defaults(func=cmd_procedure)
    pl = proc_sub.add_parser("list", help="List stored procedures")
    pl.add_argument("--agent", default=None)
    pl.add_argument("--limit", type=int, default=20)
    pl.set_defaults(func=cmd_procedure)

    p = sub.add_parser("error", help="Manage error/lesson memory")
    err_sub = p.add_subparsers(dest="error_action", required=True)
    pa = err_sub.add_parser("add", help="Add a mistake/fix/lesson memory")
    pa.add_argument("--mistake", required=True)
    pa.add_argument("--fix", default=None)
    pa.add_argument("--lesson", default=None)
    pa.add_argument("--agent", default=None)
    pa.add_argument("--task", default=None)
    pa.set_defaults(func=cmd_error)
    pl = err_sub.add_parser("list", help="List error/lesson memories")
    pl.add_argument("--agent", default=None)
    pl.add_argument("--query", default=None)
    pl.add_argument("--limit", type=int, default=10)
    pl.set_defaults(func=cmd_error)

    p = sub.add_parser("update-check", help="Check if a newer version is available")
    p.set_defaults(func=cmd_update_check)

    p = sub.add_parser("version", help="Show installed version")
    p.set_defaults(func=cmd_version)

    # Models
    p = sub.add_parser("models", help="Manage embedding models")
    model_sub = p.add_subparsers(dest="model_action", required=True)

    p_list = model_sub.add_parser("list", help="Show available models")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_models)

    p_rec = model_sub.add_parser("recommend", help="Recommend a model for this machine")
    p_rec.set_defaults(func=cmd_models)

    p_down = model_sub.add_parser("download", help="Download a model")
    p_down.add_argument("--tier", choices=list(MODEL_CONFIGS.keys()), default=None)
    p_down.add_argument("--cache-folder", default=None)
    p_down.set_defaults(func=cmd_models)

    p_switch = model_sub.add_parser("switch", help="Set the default model tier")
    p_switch.add_argument("--tier", choices=list(MODEL_CONFIGS.keys()), required=True)
    p_switch.add_argument("--download", action="store_true")
    p_switch.add_argument("--cache-folder", default=None)
    p_switch.set_defaults(func=cmd_models)

    p_migrate = model_sub.add_parser("migrate", help="Re-embed memories into a new tier")
    p_migrate.add_argument("--from-tier", choices=list(MODEL_CONFIGS.keys()), required=True)
    p_migrate.add_argument("--to-tier", choices=list(MODEL_CONFIGS.keys()), required=True)
    p_migrate.add_argument("--download", action="store_true")
    p_migrate.add_argument("--cache-folder", default=None)
    p_migrate.add_argument("--limit", type=int, default=None)
    p_migrate.add_argument("--consolidate", action="store_true")
    p_migrate.set_defaults(func=cmd_models)

    # Cross-agent communication
    p = sub.add_parser("share", help="Share useful task memory with other agents")
    share_sub = p.add_subparsers(dest="share_action", required=True)
    ps = share_sub.add_parser("send", help="Send a shared memory note to one or more agents")
    ps.add_argument("content_pos", nargs="?", help="Shared note text")
    ps.add_argument("--content", "-c", default=None)
    ps.add_argument("--from-agent", default=None, help="Sender agent name; defaults to --agent-id")
    ps.add_argument("--to", action="append", required=True, help="Recipient agent; repeat for multiple agents")
    ps.add_argument("--task", default=None, help="Task/project this shared memory belongs to")
    ps.add_argument("--importance", type=float, default=0.75)
    ps.add_argument("--priority", default="high", choices=["low", "normal", "high", "critical"])
    ps.add_argument("--tags", default="")
    ps.set_defaults(func=cmd_share)

    pi = share_sub.add_parser("inbox", help="List shared memories received by an agent")
    pi.add_argument("--agent", default=None)
    pi.add_argument("--query", default=None)
    pi.add_argument("--task", default=None)
    pi.add_argument("--limit", type=int, default=20)
    pi.set_defaults(func=cmd_share)

    po = share_sub.add_parser("outbox", help="List shared memories sent by an agent")
    po.add_argument("--agent", default=None)
    po.add_argument("--query", default=None)
    po.add_argument("--task", default=None)
    po.add_argument("--limit", type=int, default=20)
    po.set_defaults(func=cmd_share)

    pl = share_sub.add_parser("list", help="List all shared memories visible to an agent")
    pl.add_argument("--agent", default=None)
    pl.add_argument("--query", default=None)
    pl.add_argument("--task", default=None)
    pl.add_argument("--limit", type=int, default=20)
    pl.set_defaults(func=cmd_share)

    # Hook
    p = sub.add_parser("hook", help="OpenClaw hook entrypoint")
    p.add_argument("event",
                   choices=["receive", "before-reply", "preprocessed", "send", "tool-result"])
    p.add_argument("--message", default="")
    p.add_argument("--query", default="")
    p.add_argument("--role", default="user")
    p.add_argument("--tool-name", default="tool")
    p.add_argument("--importance", type=float, default=0.5)
    p.add_argument("--time-context", default="current",
                   choices=["past", "current", "future", "timeless"])
    p.add_argument("--max-tokens", type=int, default=1000)
    p.set_defaults(func=cmd_hook)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
