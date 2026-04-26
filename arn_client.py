#!/usr/bin/env python3
"""
arn_client.py — Native socket client for ARN daemon.
Bypasses subprocess overhead for ~5-10x faster recall.

Falls back to direct CLI if daemon is not running.

Usage (as script):
  python3 arn_client.py recall "what port is dco-site"
  python3 arn_client.py context "DCO website architecture"
  python3 arn_client.py store "some fact" --importance 0.8
  python3 arn_client.py stats
  python3 arn_client.py ping

Usage (as module):
  from arn_client import recall, store, perceive, context, stats
  hits = recall("what port is dco-site", top_k=3)
  store("dco runs on 4173", importance=0.9)
"""

import sys
import os
import json
import socket
import time
import subprocess
from typing import Optional

SOCKET_PATH = "/tmp/arn_daemon.sock"
ARN_CLI     = "/home/mokali/arn_v9/arn/phase2/arn_cli.py"
TIMEOUT     = 25  # seconds

# ── Core socket call ────────────────────────────────────────────────────────

def _daemon_call(req: dict, timeout: int = TIMEOUT) -> Optional[dict]:
    """Send request to daemon via Unix socket. Returns None if daemon unavailable."""
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)
        sock.sendall((json.dumps(req) + "\n").encode())
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        sock.close()
        resp = json.loads(data.decode().strip())
        if resp.get("status") == "ok":
            return resp.get("result")
        return None
    except Exception:
        return None

def _cli_fallback(args: list) -> str:
    """Fall back to direct CLI call if daemon unavailable."""
    r = subprocess.run(
        ["python3", ARN_CLI] + args,
        capture_output=True, text=True, timeout=60
    )
    return r.stdout

# ── Public API ──────────────────────────────────────────────────────────────

def recall(query: str, top_k: int = 5) -> list:
    """Recall memories matching query. Returns list of hit dicts."""
    result = _daemon_call({"cmd": "recall", "query": query, "top_k": top_k})
    if result is not None:
        return result if isinstance(result, list) else []
    # Fallback
    out = _cli_fallback(["recall", "--query", query, "--top-k", str(top_k)])
    try:
        return json.loads(out)
    except:
        return []

def context(query: str, max_tokens: int = 1200) -> str:
    """Get formatted context block for session injection."""
    result = _daemon_call({"cmd": "context", "query": query, "max_tokens": max_tokens})
    if result is not None:
        return str(result)
    return _cli_fallback(["context", "--query", query, "--max-tokens", str(max_tokens)])

def perceive(content: str, source: str = "developer") -> dict:
    """Passively absorb content into ARN episodic memory."""
    result = _daemon_call({"cmd": "perceive", "content": content, "source": source})
    if result is not None:
        return result if isinstance(result, dict) else {"stored": True}
    out = _cli_fallback(["perceive", "--content", content, "--source", source])
    try:
        return json.loads(out)
    except:
        return {"stored": False, "error": "parse failed"}

def store(content: str, importance: float = 0.7, source: str = "developer",
          time_context: str = "current") -> dict:
    """Store a high-importance memory explicitly."""
    result = _daemon_call({
        "cmd": "store", "content": content,
        "importance": importance, "source": source,
        "time_context": time_context
    })
    if result is not None:
        return result if isinstance(result, dict) else {"stored": True}
    out = _cli_fallback([
        "store", "--content", content,
        "--importance", str(importance),
        "--source", source,
        "--time-context", time_context
    ])
    try:
        return json.loads(out)
    except:
        return {"stored": False, "error": "parse failed"}

def stats() -> dict:
    """Get ARN system stats."""
    result = _daemon_call({"cmd": "stats"})
    if result is not None:
        return result if isinstance(result, dict) else {}
    out = _cli_fallback(["stats"])
    try:
        return json.loads(out)
    except:
        return {}

def ping() -> bool:
    """Check if daemon is alive."""
    result = _daemon_call({"cmd": "ping"}, timeout=3)
    return result == "pong" or (isinstance(result, str) and "pong" in result)

def daemon_latency() -> float:
    """Measure round-trip latency to daemon in seconds."""
    t0 = time.time()
    _daemon_call({"cmd": "ping"}, timeout=5)
    return round(time.time() - t0, 4)

# ── CLI entry ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    t0 = time.time()

    if cmd == "ping":
        alive = ping()
        lat = daemon_latency()
        print(json.dumps({"daemon": "up" if alive else "down", "latency_ms": round(lat*1000, 1)}))

    elif cmd == "recall":
        query = " ".join(sys.argv[2:])
        hits = recall(query)
        print(json.dumps(hits, indent=2))
        elapsed = round(time.time() - t0, 3)
        print(f"\n# {len(hits)} hits in {elapsed}s", file=sys.stderr)

    elif cmd == "context":
        query = " ".join(sys.argv[2:])
        print(context(query))
        elapsed = round(time.time() - t0, 3)
        print(f"\n# context in {elapsed}s", file=sys.stderr)

    elif cmd == "store":
        if len(sys.argv) < 3:
            print("Usage: arn_client.py store '<content>' [importance]")
            sys.exit(1)
        content = sys.argv[2]
        importance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        r = store(content, importance)
        print(json.dumps(r))

    elif cmd == "perceive":
        content = " ".join(sys.argv[2:])
        r = perceive(content)
        print(json.dumps(r))

    elif cmd == "stats":
        r = stats()
        print(json.dumps(r, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
