#!/usr/bin/env python3
"""
ARN v9 CLI — command-line interface for OpenClaw skill integration.

Called by the arn-memory SKILL.md instructions. Each subcommand maps
to an ARNPlugin method and prints JSON to stdout for the agent to parse.

Usage:
    python3 arn_cli.py store --content "fact to remember" --importance 0.8
    python3 arn_cli.py recall --query "what do I know about X?" --top-k 5
    python3 arn_cli.py context --query "current topic" --max-tokens 1000
    python3 arn_cli.py maintain
    python3 arn_cli.py stats

Environment:
    OPENCLAW_AGENT_ID   Agent namespace (default: "default")
    ARN_DATA_ROOT       Storage directory (default: ~/.arn_data)
"""

import sys
import os
import json
import argparse

# Ensure arn package is importable
# Try: parent of the script's grandparent (arn/scripts/ → arn_v9/ → parent)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_dir = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _package_dir)
# Also try home directory (standard Pi deployment)
sys.path.insert(0, os.path.expanduser('~'))

from arn.plugin import ARNPlugin

DEFAULT_DATA_ROOT = os.environ.get(
    'ARN_DATA_ROOT',
    os.path.expanduser('~/.arn_data')
)
DEFAULT_AGENT_ID = os.environ.get('OPENCLAW_AGENT_ID', 'default')


def get_plugin(strict: bool = False):
    """
    Create a plugin instance. Each CLI invocation opens and closes cleanly.
    
    If strict=True and embeddings are degraded (model missing), prints
    an error to stderr and exits with code 1 instead of returning garbage.
    """
    plugin = ARNPlugin(
        agent_id=DEFAULT_AGENT_ID,
        data_root=DEFAULT_DATA_ROOT,
        auto_consolidate=True,
        consolidation_threshold=128,
    )
    
    if plugin._arn.embedder.is_degraded:
        msg = (
            "ERROR: Embedding model is not loaded. "
            "ARN cannot function without semantic embeddings.\n"
            "All recall/store operations will return meaningless results.\n"
            "Fix: pip install sentence-transformers\n"
            "Then ensure internet access for first model download, or copy\n"
            "~/.cache/huggingface/hub/ from a machine that has the model."
        )
        if strict:
            print(msg, file=sys.stderr)
            plugin.shutdown()
            sys.exit(1)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
    
    return plugin


def cmd_perceive(args):
    """Passively absorb text — auto-detects importance, never stores credentials."""
    import sys as _sys
    _sys.path.insert(0, '/home/mokali/arn')
    from arn.memory_llm import detect_importance, detect_time_context
    importance = detect_importance(args.content)
    if importance == 0.0:
        print(json.dumps({"stored": False, "reason": "credential pattern detected — not stored"}))
        return
    if importance < args.min_importance:
        print(json.dumps({"stored": False, "reason": f"importance {importance:.2f} below threshold {args.min_importance}"}))
        return
    time_ctx = args.time_context if args.time_context != 'auto' else detect_time_context(args.content)
    with get_plugin(strict=False) as plugin:
        result = plugin.store(
            content=args.content,
            importance=importance,
            source=args.source,
            time_context=time_ctx,
        )
        result['auto_importance'] = importance
        print(json.dumps(result, indent=2))


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
    """Recall relevant memories for a query."""
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


def cmd_maintain(args):
    """Run consolidation and maintenance."""
    with get_plugin(strict=args.strict) as plugin:
        stats = plugin.maintain()
        print(json.dumps(stats, indent=2))


def cmd_stats(args):
    """Print system statistics."""
    with get_plugin(strict=args.strict) as plugin:
        stats = plugin.get_stats()
        print(json.dumps(stats, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description='ARN v9 Memory CLI for OpenClaw',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s store --content "User prefers Python" --importance 0.8 --tags "preference"
  %(prog)s recall --query "programming language preference" --top-k 3
  %(prog)s context --query "help with code" --max-tokens 500
  %(prog)s maintain
  %(prog)s stats
  %(prog)s --strict store --content "fails if no model" --importance 0.5
        """
    )
    parser.add_argument('--strict', action='store_true', default=False,
                        help='Exit with error code 1 if embedding model is unavailable')
    sub = parser.add_subparsers(dest='command')
    sub.required = True

    # perceive (passive auto-absorb)
    p_perceive = sub.add_parser('perceive', help='Passively absorb text with auto-importance detection')
    p_perceive.add_argument('--content', '-c', required=True, help='Text to absorb')
    p_perceive.add_argument('--source', '-s', default='user', help='Source label (default: user)')
    p_perceive.add_argument('--time-context', default='auto',
                            choices=['auto', 'past', 'current', 'future'],
                            help='Temporal context — auto detects from text (default: auto)')
    p_perceive.add_argument('--min-importance', type=float, default=0.15,
                            help='Skip storage if auto-detected importance is below this (default: 0.15)')
    p_perceive.set_defaults(func=cmd_perceive)

    # store
    p_store = sub.add_parser('store', help='Store a new memory')
    p_store.add_argument('--content', '-c', required=True,
                         help='Text content to remember')
    p_store.add_argument('--importance', '-i', type=float, default=0.5,
                         help='Importance score 0.0-1.0 (default: 0.5)')
    p_store.add_argument('--tags', '-t', default='',
                         help='Comma-separated tags (e.g., "preference,project")')
    p_store.add_argument('--source', '-s', default='agent',
                         help='Source of the memory (default: agent)')
    p_store.add_argument('--time-context', default='current',
                         choices=['past', 'current', 'future'],
                         help='Temporal scope (default: current). Use "past" for '
                              '"used to" facts, "future" for "will" facts.')
    p_store.set_defaults(func=cmd_store)

    # recall
    p_recall = sub.add_parser('recall', help='Recall relevant memories')
    p_recall.add_argument('--query', '-q', required=True,
                          help='Natural language query')
    p_recall.add_argument('--top-k', '-k', type=int, default=5,
                          help='Number of results (default: 5)')
    p_recall.add_argument('--time-filter', default=None,
                          choices=['past', 'current', 'future'],
                          help='Explicit temporal filter (default: auto-detect from query)')
    p_recall.set_defaults(func=cmd_recall)

    # context
    p_ctx = sub.add_parser('context', help='Get formatted context for prompts')
    p_ctx.add_argument('--query', '-q', default='',
                       help='Optional query to focus retrieval')
    p_ctx.add_argument('--max-tokens', '-m', type=int, default=1000,
                       help='Token budget for context (default: 1000)')
    p_ctx.set_defaults(func=cmd_context)

    # maintain
    p_maint = sub.add_parser('maintain', help='Run memory consolidation')
    p_maint.set_defaults(func=cmd_maintain)

    # stats
    p_stats = sub.add_parser('stats', help='Show system statistics')
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
