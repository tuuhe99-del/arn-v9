#!/usr/bin/env python3
"""
ARN v9 CLI
===========
Persistent memory from the command line. Works with any AI coding assistant:
Codex, Claude Code, Kimi, Aider, OpenClaw, or plain terminal.

Commands:
    arn store   -c "fact to remember" -i 0.8
    arn recall  -q "what do I know?" -k 5
    arn context -q "current topic" -m 1000
    arn forget  -q "topic to forget"
    arn stats
    arn maintain
    arn setup   [--tier nano|base] [--client codex|claude|kimi|openclaw]
    arn export  -o backup.json
    arn import  -f backup.json

Environment (all optional, sensible defaults):
    ARN_DATA_DIR          Storage directory (default: ~/.arn_data)
    ARN_EMBEDDING_TIER    Model tier: nano|small|base|base-e5 (default: nano)
    ARN_AGENT_ID          Agent namespace (default: default)

Bug fixes over v9.0:
    - Uses ARN_EMBEDDING_TIER not ARN_EMBEDDING_MODEL (was TypeError)
    - Consistent data directory (always ARN_DATA_DIR, no more cli vs default)
    - Suppresses HuggingFace unauthenticated warnings
    - Model pre-download in setup avoids degraded-mode surprise
"""

import sys
import os
import json
import argparse
import hashlib
import time
import logging
import warnings
from pathlib import Path
import shutil

# ─── Suppress noisy warnings BEFORE any imports ───
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', message='.*Unauthenticated.*')
warnings.filterwarnings('ignore', message='.*huggingface.*')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

# ─── Make arn_v9 importable from any install location ───
_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parent.parent  # arn_v9/scripts → arn_v9 → root
sys.path.insert(0, str(_package_root))
# Fallback: check if installed under ~/
sys.path.insert(0, str(Path.home()))


# ─── Resolve config from env with correct variable names ───
def get_config():
    """
    Single source of truth for all configuration.
    Reads env vars with correct names, provides sensible defaults.
    """
    # ARN_DATA_DIR is the canonical var. Also check legacy ARN_DATA_ROOT.
    data_dir = os.environ.get(
        'ARN_DATA_DIR',
        os.environ.get('ARN_DATA_ROOT', str(Path.home() / '.arn_data'))
    )
    
    # ARN_EMBEDDING_TIER is the canonical var. 
    # If someone set ARN_EMBEDDING_MODEL, translate it.
    tier = os.environ.get('ARN_EMBEDDING_TIER', None)
    if tier is None:
        model = os.environ.get('ARN_EMBEDDING_MODEL', '')
        tier = _model_to_tier(model) if model else 'nano'
    
    agent_id = os.environ.get(
        'ARN_AGENT_ID',
        os.environ.get('OPENCLAW_AGENT_ID', 'default')
    )
    
    return {
        'data_dir': data_dir,
        'tier': tier,
        'agent_id': agent_id,
    }


def _model_to_tier(model_name: str) -> str:
    """Translate a model name to the correct tier string."""
    mapping = {
        'all-MiniLM-L6-v2': 'nano',
        'sentence-transformers/all-MiniLM-L6-v2': 'nano',
        'all-mpnet-base-v2': 'small',
        'sentence-transformers/all-mpnet-base-v2': 'small',
        'bge-base-en-v1.5': 'base',
        'BAAI/bge-base-en-v1.5': 'base',
        'e5-base-v2': 'base-e5',
        'intfloat/e5-base-v2': 'base-e5',
    }
    for key, tier in mapping.items():
        if key in model_name:
            return tier
    return 'nano'  # Safe default


# ─── Plugin factory ───
def get_plugin(strict: bool = False, config: dict = None):
    """
    Create a plugin instance with correct config.
    Always uses ARN_EMBEDDING_TIER (not model name) and
    consistent data directory.
    """
    from arn_v9.plugin import ARNPlugin
    
    if config is None:
        config = get_config()
    
    plugin = ARNPlugin(
        agent_id=config['agent_id'],
        data_root=config['data_dir'],
        embedding_tier=config['tier'],
        auto_consolidate=True,
        consolidation_threshold=128,
    )
    
    if plugin._arn.embedder.is_degraded:
        msg = (
            "Embedding model not loaded. Memory will not work correctly.\n"
            "Fix: pip install sentence-transformers\n"
            "Then run: arn setup"
        )
        if strict:
            print(msg, file=sys.stderr)
            plugin.shutdown()
            sys.exit(1)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
    
    return plugin


# ═══════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════

def cmd_store(args):
    """Store a new memory."""
    with get_plugin(strict=args.strict) as plugin:
        tags = [t.strip() for t in args.tags.split(',') if t.strip()] if args.tags else []
        result = plugin.store(
            content=args.content,
            importance=args.importance,
            tags=tags,
            source=args.source,
            time_context=args.time_context,
        )
        print(json.dumps(result, indent=2))


def cmd_recall(args):
    """Recall relevant memories."""
    with get_plugin(strict=args.strict) as plugin:
        results = plugin.recall(
            query=args.query,
            top_k=args.top_k,
            time_filter=args.time_filter,
        )
        print(json.dumps(results, indent=2))


def cmd_context(args):
    """Get formatted context window for prompt injection."""
    with get_plugin(strict=args.strict) as plugin:
        context = plugin.get_context_window(
            query=args.query if args.query else None,
            max_tokens=args.max_tokens,
        )
        print(context)


def cmd_forget(args):
    """Forget memories matching a query."""
    with get_plugin(strict=args.strict) as plugin:
        results = plugin.recall(query=args.query, top_k=args.top_k)
        strong = [r for r in results
                  if r.get('similarity', 0) >= args.min_similarity]
        
        if not strong:
            print(json.dumps({"forgotten": 0, "message": "No matching memories found"}))
            return
        
        ids = [r['id'] for r in strong if r.get('type') == 'episodic']
        if ids:
            plugin._arn.storage.delete_episodes(ids)
        
        print(json.dumps({
            "forgotten": len(ids),
            "matched": [r['content'][:80] for r in strong],
        }, indent=2))


def cmd_maintain(args):
    """Run consolidation and maintenance."""
    with get_plugin(strict=args.strict) as plugin:
        stats = plugin.maintain()
        print(json.dumps(stats, indent=2))


def cmd_stats(args):
    """Print system statistics."""
    with get_plugin(strict=args.strict) as plugin:
        stats = plugin.get_stats()
        config = get_config()
        stats['config'] = config
        print(json.dumps(stats, indent=2, default=str))


def cmd_export(args):
    """Export all memories to JSON."""
    with get_plugin(strict=args.strict) as plugin:
        episodes = plugin._arn.storage.get_all_episodes()
        export = {
            'version': 'arn_v9_export_v1',
            'exported_at': time.time(),
            'agent_id': get_config()['agent_id'],
            'episode_count': len(episodes),
            'episodes': episodes,
        }
        
        outpath = args.output or 'arn_backup.json'
        with open(outpath, 'w') as f:
            json.dump(export, f, indent=2, default=str)
        
        print(json.dumps({"exported": len(episodes), "file": outpath}))


def cmd_import(args):
    """Import memories from JSON."""
    with get_plugin(strict=args.strict) as plugin:
        with open(args.file, 'r') as f:
            data = json.load(f)
        
        imported = 0
        skipped = 0
        for ep in data.get('episodes', []):
            content = ep.get('content', '')
            if not content:
                continue
            try:
                plugin.store(
                    content=content,
                    importance=ep.get('importance', 0.5),
                    source=ep.get('source', 'import'),
                )
                imported += 1
            except Exception:
                skipped += 1
        
        print(json.dumps({
            "imported": imported,
            "skipped": skipped,
            "total_in_file": len(data.get('episodes', [])),
        }, indent=2))


# ═══════════════════════════════════════════
# SETUP COMMAND
# ═══════════════════════════════════════════

# Client instruction templates
_CLIENT_INSTRUCTIONS = {
    'codex': {
        'filename': 'arn.md',
        'paths': [
            Path.home() / '.codex' / 'memories',
            Path.home() / '.codex',
        ],
        'content': """# ARN Memory

You have persistent memory via ARN. Use it to remember facts across sessions.

## Store a fact
```bash
arn store -c "fact to remember" -i 0.8 --time-context current
```
Importance: 0.9 for identity/names, 0.7-0.8 for preferences/decisions, 0.5 for general facts, 0.1-0.2 for chitchat.
Time context: "current" for present facts, "past" for "used to" facts, "future" for plans.

## Recall memories
```bash
arn recall -q "what is relevant to this topic" -k 5
```
Check the `confidence_tier` field: high = state it, medium = hedge, low = don't use it.

## Get context for your prompt
```bash
arn context -q "current conversation topic" -m 1000
```

## Rules
- Store important user facts immediately
- Recall before answering questions about past discussions
- Never store passwords, API keys, or credentials
- Run `arn maintain` during idle time
""",
    },
    'claude': {
        'filename': 'arn-memory.md',
        'paths': [
            Path.home() / '.claude',
            Path.home() / '.claude' / 'memories',
        ],
        'content': """# ARN Memory Integration

You have persistent memory through the `arn` CLI. It survives across sessions.

## Commands
- `arn store -c "fact" -i 0.8` — store a fact (importance 0.0-1.0)
- `arn recall -q "query" -k 5` — find relevant memories
- `arn context -q "topic"` — get formatted context block
- `arn forget -q "topic"` — delete memories about a topic
- `arn stats` — check memory status

## When to use
- **Store** when the user shares identity, preferences, decisions, or project facts
- **Recall** before answering questions that reference previous conversations
- **Context** when you need comprehensive background for a complex task

## Confidence tiers
Each recall result has a `confidence_tier`:
- `high` — state the fact plainly
- `medium` — say "I believe you mentioned..."
- `low` — don't use this result, it's a weak match

## Important
- Never store credentials (passwords, API keys, tokens)
- Use `--time-context past` for "used to" facts, `future` for plans
- Run `arn maintain` periodically to consolidate memories
""",
    },
    'kimi': {
        'filename': 'arn-memory.md',
        'paths': [
            Path.home() / '.kimi',
            Path.home() / '.kimi' / 'memories',
        ],
        'content': """# ARN Memory

Persistent memory via `arn` CLI. Store facts, recall context across sessions.

## Quick reference
```bash
arn store -c "user's name is Mohamed" -i 0.9
arn recall -q "user's name" -k 3
arn context -q "current topic" -m 1000
arn forget -q "outdated information"
arn stats
arn maintain
```

## Rules
- Store important facts immediately (names=0.9, preferences=0.7, general=0.5)
- Recall before answering history-dependent questions
- Check confidence_tier: high=certain, medium=hedge, low=ignore
- Never store passwords or API keys
""",
    },
    'openclaw': {
        'filename': 'SKILL.md',
        'paths': [
            Path.home() / '.openclaw' / 'skills' / 'arn-memory',
        ],
        'content': None,  # Uses the full SKILL.md from the package
    },
    'aider': {
        'filename': '.aider.conf.yml',
        'paths': [
            Path.home(),
        ],
        'content': """# ARN Memory Integration for Aider
# Add this to your aider conventions or read it at session start

# Before answering questions about past work, run:
#   arn recall -q "relevant topic" -k 5
# After learning important facts, run:
#   arn store -c "fact" -i 0.8
# For full context injection:
#   arn context -q "current topic" -m 1000
""",
    },
}


def cmd_setup(args):
    """
    One-command setup for ARN + AI client integration.
    
    Handles:
    1. Dependency verification
    2. Data directory creation
    3. Environment variables (persistent via ~/.bashrc)
    4. Model download and verification
    5. Store/recall round-trip test
    6. Client-specific instruction files
    """
    tier = args.tier
    client = args.client
    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / '.arn_data'
    
    print(f"\nARN v9 Setup")
    print(f"  Tier:   {tier}")
    print(f"  Client: {client or 'none'}")
    print(f"  Data:   {data_dir}\n")
    
    # Step 1: Check dependencies
    print("Checking dependencies...")
    
    missing = []
    try:
        import numpy
        print(f"  numpy: ok")
    except ImportError:
        missing.append('numpy')
        print(f"  numpy: MISSING")
    
    try:
        import sentence_transformers
        print(f"  sentence-transformers: ok")
    except ImportError:
        missing.append('sentence-transformers')
        print(f"  sentence-transformers: MISSING")
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\nThen re-run: arn setup --tier {tier}")
        sys.exit(1)
    
    # Optional deps
    for pkg, label in [('rank_bm25', 'BM25 search'), ('spacy', 'entity extraction')]:
        try:
            __import__(pkg)
            print(f"  {label}: ok")
        except ImportError:
            print(f"  {label}: not installed (optional)")
    
    # Step 2: Create data directory
    print(f"\nSetting up directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'default').mkdir(exist_ok=True)
    print(f"  Created: {data_dir}")
    
    # Step 3: Set environment variables persistently
    print(f"\nConfiguring environment...")
    env_lines = {
        'ARN_DATA_DIR': str(data_dir),
        'ARN_EMBEDDING_TIER': tier,
        'ARN_AGENT_ID': 'default',
    }
    
    bashrc = Path.home() / '.bashrc'
    if bashrc.exists():
        content = bashrc.read_text()
        additions = []
        for var, val in env_lines.items():
            # Remove any existing ARN env lines first
            lines = content.split('\n')
            lines = [l for l in lines if not l.strip().startswith(f'export {var}=')]
            content = '\n'.join(lines)
            additions.append(f'export {var}="{val}"')
        
        with open(bashrc, 'w') as f:
            f.write(content.rstrip() + '\n\n# ARN v9 configuration\n')
            f.write('\n'.join(additions) + '\n')
        
        print(f"  Updated ~/.bashrc")
    
    # Set for current process too
    for var, val in env_lines.items():
        os.environ[var] = val
        print(f"  {var}={val}")
    
    # Step 4: Download model
    print(f"\nDownloading embedding model ({tier})...")
    print(f"  This may take 1-2 minutes on first run...")
    
    try:
        from arn_v9.core.embeddings import EmbeddingEngine, MODEL_CONFIGS
        model_info = MODEL_CONFIGS.get(tier, MODEL_CONFIGS['nano'])
        print(f"  Model: {model_info['name']}")
        
        engine = EmbeddingEngine(use_model=True, tier=tier)
        if engine.is_degraded:
            print(f"  WARNING: Model failed to load. Check internet connection.")
            print(f"  ARN will retry on next use.")
        else:
            test_vec = engine.encode("test")
            print(f"  Loaded: {test_vec.shape[0]}-dimensional vectors")
            print(f"  Status: ready")
    except Exception as e:
        print(f"  Model download issue: {e}")
        print(f"  ARN will retry on first use.")
    
    # Step 5: Test store/recall
    print(f"\nTesting memory...")
    try:
        from arn_v9.plugin import ARNPlugin
        plugin = ARNPlugin(
            agent_id='default',
            data_root=str(data_dir),
            embedding_tier=tier,
        )
        
        plugin.store(
            content="ARN setup test — this will be deleted",
            importance=0.5,
            source='setup',
        )
        
        results = plugin.recall("ARN setup test", top_k=1)
        if results and 'setup test' in results[0].get('content', ''):
            print(f"  Store:  ok")
            print(f"  Recall: ok (confidence: {results[0].get('confidence_tier', 'unknown')})")
            # Clean up test memory
            if results[0].get('type') == 'episodic':
                plugin._arn.storage.delete_episodes([results[0]['id']])
        else:
            print(f"  Store:  ok")
            print(f"  Recall: returned different result (model may still be loading)")
        
        plugin.shutdown()
    except Exception as e:
        print(f"  Test failed: {e}")
    
    # Step 6: Detect and connect clients
    if client:
        print(f"\nConnecting to {client}...")
        _setup_client(client, tier)
    else:
        print(f"\nDetecting AI clients...")
        detected = _detect_clients()
        
        if not detected:
            print("  No AI clients detected.")
            print("  Supported: Claude Code, Codex CLI, OpenClaw, Kimi, Aider")
            print("  Install one, then run: arn setup --client <name>")
        elif len(detected) == 1:
            name, info = detected[0]
            print(f"  Found: {info['display']} ({info['evidence']})")
            _setup_client(name, tier)
        else:
            print(f"  Found {len(detected)} clients:")
            for i, (name, info) in enumerate(detected, 1):
                print(f"    {i}. {info['display']} ({info['evidence']})")
            print(f"\n  Connecting all detected clients...")
            for name, info in detected:
                _setup_client(name, tier)
    
    # Done
    print(f"\n{'='*50}")
    print(f"ARN memory is ready.")
    print(f"  Model:  {tier}")
    print(f"  Data:   {data_dir}")
    print(f"  Agent:  default")
    print(f"\nQuick test:")
    print(f'  arn store -c "My name is Mohamed" -i 0.9')
    print(f'  arn recall -q "what is my name" -k 1')
    print(f"\nRestart your terminal or run: source ~/.bashrc")
    print()


def _setup_client(client: str, tier: str):
    """Write client-specific instruction files."""
    template = _CLIENT_INSTRUCTIONS.get(client)
    if not template:
        print(f"  Unknown client: {client}")
        print(f"  Supported: codex, claude, kimi, openclaw")
        return
    
    # Find or create the target directory
    target_dir = None
    for p in template['paths']:
        if p.exists():
            target_dir = p
            break
    
    if target_dir is None:
        # Create the first path option
        target_dir = template['paths'][0]
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {target_dir}")
    
    # Write the instruction file
    if client == 'openclaw':
        # Copy the full SKILL.md from the package
        skill_src = _script_dir.parent / 'openclaw_skill' / 'SKILL.md'
        if skill_src.exists():
            import shutil
            target_file = target_dir / 'SKILL.md'
            shutil.copy2(str(skill_src), str(target_file))
            
            # Fix CLI paths in the copied SKILL.md
            content = target_file.read_text()
            cli_path = str(_script_dir / 'arn_cli.py')
            content = content.replace(
                '~/arn_v9/scripts/arn_cli.py',
                cli_path
            )
            target_file.write_text(content)
            print(f"  Wrote: {target_file}")
        else:
            print(f"  SKILL.md not found at {skill_src}")
    else:
        content = template['content']
        target_file = target_dir / template['filename']
        target_file.write_text(content)
        print(f"  Wrote: {target_file}")
    
    print(f"  {client} integration configured")


def _detect_clients() -> list:
    """
    Auto-detect which AI CLI clients are installed.
    Checks for binaries in PATH and config directories.
    Returns list of (name, info_dict) tuples.
    """
    detected = []
    
    # Claude Code
    for evidence_path, description in [
        (shutil.which('claude'), 'claude binary in PATH'),
        (Path.home() / '.claude', '~/.claude directory exists'),
    ]:
        if evidence_path and (isinstance(evidence_path, str) or evidence_path.exists()):
            detected.append(('claude', {'display': 'Claude Code', 'evidence': description}))
            break
    
    # Codex CLI
    for evidence_path, description in [
        (shutil.which('codex'), 'codex binary in PATH'),
        (Path.home() / '.codex', '~/.codex directory exists'),
    ]:
        if evidence_path and (isinstance(evidence_path, str) or evidence_path.exists()):
            detected.append(('codex', {'display': 'Codex CLI', 'evidence': description}))
            break
    
    # OpenClaw
    for evidence_path, description in [
        (shutil.which('openclaw'), 'openclaw binary in PATH'),
        (Path.home() / '.openclaw', '~/.openclaw directory exists'),
    ]:
        if evidence_path and (isinstance(evidence_path, str) or evidence_path.exists()):
            detected.append(('openclaw', {'display': 'OpenClaw', 'evidence': description}))
            break
    
    # Kimi
    if (Path.home() / '.kimi').exists():
        detected.append(('kimi', {'display': 'Kimi', 'evidence': '~/.kimi directory exists'}))
    
    # Aider
    if shutil.which('aider'):
        detected.append(('aider', {'display': 'Aider', 'evidence': 'aider binary in PATH'}))
    
    return detected


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='arn',
        description='ARN v9 — Persistent memory for AI agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  arn setup --tier nano --client codex
  arn store -c "User prefers Python" -i 0.8
  arn recall -q "programming preferences" -k 3
  arn stats

Clients: codex, claude, kimi, openclaw
Tiers:   nano (22MB, fast), small (420MB), base (440MB, best), base-e5 (440MB)
        """
    )
    parser.add_argument('--strict', action='store_true', default=False,
                        help='Exit with error if embedding model unavailable')
    
    sub = parser.add_subparsers(dest='command')
    sub.required = True
    
    # ─── setup ───
    p_setup = sub.add_parser('setup', help='One-command setup and integration')
    p_setup.add_argument('--tier', '-t', default='nano',
                         choices=['nano', 'small', 'base', 'base-e5'],
                         help='Embedding model tier (default: nano)')
    p_setup.add_argument('--client', '-c', default=None,
                         choices=['codex', 'claude', 'kimi', 'openclaw', 'aider'],
                         help='AI client to integrate with (auto-detects if omitted)')
    p_setup.add_argument('--data-dir', '-d', default=None,
                         help=f'Data directory (default: ~/.arn_data)')
    p_setup.set_defaults(func=cmd_setup)
    
    # ─── store ───
    p_store = sub.add_parser('store', help='Store a memory')
    p_store.add_argument('--content', '-c', required=True,
                         help='Text to remember')
    p_store.add_argument('--importance', '-i', type=float, default=0.5,
                         help='Importance 0.0-1.0 (default: 0.5)')
    p_store.add_argument('--tags', default='',
                         help='Comma-separated tags')
    p_store.add_argument('--source', '-s', default='agent',
                         help='Source (default: agent)')
    p_store.add_argument('--time-context', default='current',
                         choices=['past', 'current', 'future'],
                         help='Temporal scope (default: current)')
    p_store.set_defaults(func=cmd_store)
    
    # ─── recall ───
    p_recall = sub.add_parser('recall', help='Recall relevant memories')
    p_recall.add_argument('--query', '-q', required=True,
                          help='Natural language query')
    p_recall.add_argument('--top-k', '-k', type=int, default=5,
                          help='Number of results (default: 5)')
    p_recall.add_argument('--time-filter', default=None,
                          choices=['past', 'current', 'future'],
                          help='Temporal filter')
    p_recall.set_defaults(func=cmd_recall)
    
    # ─── context ───
    p_ctx = sub.add_parser('context', help='Get context for prompt injection')
    p_ctx.add_argument('--query', '-q', default='',
                       help='Focus query')
    p_ctx.add_argument('--max-tokens', '-m', type=int, default=1000,
                       help='Token budget (default: 1000)')
    p_ctx.set_defaults(func=cmd_context)
    
    # ─── forget ───
    p_forget = sub.add_parser('forget', help='Forget memories about a topic')
    p_forget.add_argument('--query', '-q', required=True,
                          help='What to forget')
    p_forget.add_argument('--top-k', '-k', type=int, default=5,
                          help='Max memories to forget (default: 5)')
    p_forget.add_argument('--min-similarity', type=float, default=0.5,
                          help='Min similarity to delete (default: 0.5)')
    p_forget.set_defaults(func=cmd_forget)
    
    # ─── maintain ───
    p_maint = sub.add_parser('maintain', help='Run memory consolidation')
    p_maint.set_defaults(func=cmd_maintain)
    
    # ─── stats ───
    p_stats = sub.add_parser('stats', help='Show memory statistics')
    p_stats.set_defaults(func=cmd_stats)
    
    # ─── export ───
    p_export = sub.add_parser('export', help='Export memories to JSON')
    p_export.add_argument('--output', '-o', default='arn_backup.json',
                          help='Output file (default: arn_backup.json)')
    p_export.set_defaults(func=cmd_export)
    
    # ─── import ───
    p_import = sub.add_parser('import', help='Import memories from JSON')
    p_import.add_argument('--file', '-f', required=True,
                          help='JSON file to import')
    p_import.set_defaults(func=cmd_import)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
