#!/usr/bin/env python3
"""
ARN Daemon — persistent process that keeps embedding model loaded in RAM.
Eliminates the ~10s cold-start per CLI call by serving recall/perceive/store
requests over a Unix socket.

Socket: /tmp/arn_daemon.sock
Protocol: newline-delimited JSON

Request:  {"cmd": "recall", "query": "...", "top_k": 5}
          {"cmd": "context", "query": "...", "max_tokens": 1200}
          {"cmd": "perceive", "content": "...", "source": "developer"}
          {"cmd": "store", "content": "...", "importance": 0.8, "source": "developer"}
          {"cmd": "stats"}
          {"cmd": "ping"}
          {"cmd": "shutdown"}

Response: {"status": "ok", "result": <data>}
          {"status": "error", "message": "..."}

Usage:
  # Start daemon (background)
  python3 /home/mokali/arn_v9/arn_daemon.py start &

  # Check if running
  python3 /home/mokali/arn_v9/arn_daemon.py ping

  # Stop daemon
  python3 /home/mokali/arn_v9/arn_daemon.py stop

  # Query via daemon (fast path)
  python3 /home/mokali/arn_v9/arn_daemon.py recall "what port is dco-site"
"""

import sys
import os
import json
import socket
import signal
import time
import logging
from pathlib import Path

SOCKET_PATH = "/tmp/arn_daemon.sock"
PID_FILE    = "/tmp/arn_daemon.pid"
LOG_FILE    = "/tmp/arn_daemon.log"
ARN_ROOT    = "/home/mokali/arn_v9"
TIMEOUT     = 30  # seconds per request

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("arn_daemon")

# ── Client helpers ─────────────────────────────────────────────────────────

def send_request(req: dict, timeout: int = TIMEOUT) -> dict:
    """Send a request to the daemon and return the response."""
    if not Path(SOCKET_PATH).exists():
        return {"status": "error", "message": "Daemon not running. Start with: python3 arn_daemon.py start &"}
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
        return json.loads(data.decode().strip())
    except Exception as e:
        return {"status": "error", "message": str(e)}

def is_running() -> bool:
    if not Path(SOCKET_PATH).exists():
        return False
    r = send_request({"cmd": "ping"}, timeout=3)
    return r.get("status") == "ok"

# ── Server ─────────────────────────────────────────────────────────────────

def run_server():
    """Main daemon server loop. Keeps ARNPlugin loaded in RAM — no subprocess per call."""
    log.info("ARN Daemon starting — importing ARNPlugin directly...")

    # Add ARN to path
    sys.path.insert(0, ARN_ROOT)
    sys.path.insert(0, os.path.expanduser('~'))

    # Load ARNPlugin ONCE — this is the key: model stays in memory
    plugin = None
    try:
        from arn_v9.plugin import ARNPlugin
        plugin = ARNPlugin(
            agent_id=os.environ.get('OPENCLAW_AGENT_ID', 'default'),
            data_root=os.environ.get('ARN_DATA_ROOT', os.path.expanduser('~/.arn_data')),
            auto_consolidate=True,
            consolidation_threshold=128,
        )
        log.info(f"ARNPlugin loaded — embedder degraded: {plugin._arn.embedder.is_degraded}")
        # Warm up with a dummy query to force model load
        _ = plugin.recall("warmup", top_k=1)
        log.info("Embedding model warmed up — ready for fast recall")
    except Exception as e:
        log.error(f"Failed to load ARNPlugin: {e}. Daemon will use subprocess fallback.")
        plugin = None

    # Clean up stale socket
    if Path(SOCKET_PATH).exists():
        os.unlink(SOCKET_PATH)

    # Write PID
    Path(PID_FILE).write_text(str(os.getpid()))

    # Signal handlers
    def shutdown(sig, frame):
        log.info("Daemon shutting down")
        if Path(SOCKET_PATH).exists():
            os.unlink(SOCKET_PATH)
        if Path(PID_FILE).exists():
            os.unlink(PID_FILE)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Start socket server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    os.chmod(SOCKET_PATH, 0o600)
    log.info(f"ARN Daemon ready — socket: {SOCKET_PATH}")
    print(f"ARN Daemon ready — PID {os.getpid()} — socket: {SOCKET_PATH}", flush=True)

    while True:
        try:
            conn, _ = server.accept()
            data = b""
            conn.settimeout(TIMEOUT)
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            req = json.loads(data.decode().strip())
            cmd = req.get("cmd", "")
            log.info(f"Request: {cmd}")

            t0 = time.time()
            if cmd == "ping":
                resp = {"status": "ok", "result": "pong", "pid": os.getpid(),
                        "plugin_loaded": plugin is not None}

            elif cmd == "shutdown":
                conn.sendall((json.dumps({"status": "ok", "result": "shutting down"}) + "\n").encode())
                conn.close()
                if plugin:
                    try: plugin.shutdown()
                    except: pass
                shutdown(None, None)

            elif cmd == "recall":
                query = req.get("query", "")
                top_k = req.get("top_k", 5)
                if plugin:
                    raw = plugin.recall(query, top_k=top_k)
                    # Normalize to list of dicts
                    result = []
                    for hit in (raw or []):
                        if hasattr(hit, '__dict__'):
                            result.append(vars(hit))
                        elif isinstance(hit, dict):
                            result.append(hit)
                        else:
                            result.append({"content": str(hit)})
                else:
                    import subprocess
                    r = subprocess.run(["python3", f"{ARN_ROOT}/arn_v9/scripts/arn_cli.py",
                        "recall", "--query", query, "--top-k", str(top_k)],
                        capture_output=True, text=True, timeout=30)
                    try: result = json.loads(r.stdout)
                    except: result = []
                resp = {"status": "ok", "result": result, "latency": round(time.time()-t0, 3),
                        "via": "plugin" if plugin else "subprocess"}

            elif cmd == "context":
                query = req.get("query", "")
                max_tokens = req.get("max_tokens", 1200)
                if plugin:
                    result = plugin.get_context_window(query=query, max_tokens=max_tokens)
                else:
                    import subprocess
                    r = subprocess.run(["python3", f"{ARN_ROOT}/arn_v9/scripts/arn_cli.py",
                        "context", "--query", query, "--max-tokens", str(max_tokens)],
                        capture_output=True, text=True, timeout=30)
                    result = r.stdout
                resp = {"status": "ok", "result": result, "latency": round(time.time()-t0, 3),
                        "via": "plugin" if plugin else "subprocess"}

            elif cmd == "perceive":
                content = req.get("content", "")
                source = req.get("source", "developer")
                if plugin:
                    # perceive maps to store with auto importance
                    r2 = plugin.store(content, importance=0.3, source=source, time_context="current")
                    result = {"stored": True, "episode_id": getattr(r2, 'episode_id', None)}
                else:
                    import subprocess
                    r = subprocess.run(["python3", f"{ARN_ROOT}/arn_v9/scripts/arn_cli.py",
                        "perceive", "--content", content, "--source", source],
                        capture_output=True, text=True, timeout=30)
                    try: result = json.loads(r.stdout)
                    except: result = {"stored": False}
                resp = {"status": "ok", "result": result, "latency": round(time.time()-t0, 3)}

            elif cmd == "store":
                content = req.get("content", "")
                importance = float(req.get("importance", 0.7))
                source = req.get("source", "developer")
                time_context = req.get("time_context", "current")
                if plugin:
                    r2 = plugin.store(content, importance=importance, source=source,
                                      time_context=time_context)
                    result = {"stored": True, "episode_id": getattr(r2, 'episode_id', None)}
                else:
                    import subprocess
                    r = subprocess.run(["python3", f"{ARN_ROOT}/arn_v9/scripts/arn_cli.py",
                        "store", "--content", content, "--importance", str(importance),
                        "--source", source, "--time-context", time_context],
                        capture_output=True, text=True, timeout=30)
                    try: result = json.loads(r.stdout)
                    except: result = {"stored": False}
                resp = {"status": "ok", "result": result, "latency": round(time.time()-t0, 3)}

            elif cmd == "stats":
                if plugin:
                    result = plugin.get_stats()
                else:
                    import subprocess
                    r = subprocess.run(["python3", f"{ARN_ROOT}/arn_v9/scripts/arn_cli.py", "stats"],
                        capture_output=True, text=True, timeout=15)
                    try: result = json.loads(r.stdout)
                    except: result = {}
                resp = {"status": "ok", "result": result}

            else:
                resp = {"status": "error", "message": f"Unknown command: {cmd}"}

            log.info(f"Response: {cmd} in {time.time()-t0:.2f}s")
            conn.sendall((json.dumps(resp) + "\n").encode())
            conn.close()

        except Exception as e:
            log.error(f"Handler error: {e}")
            try:
                conn.sendall((json.dumps({"status": "error", "message": str(e)}) + "\n").encode())
                conn.close()
            except:
                pass

# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    subcmd = sys.argv[1]

    if subcmd == "start":
        if is_running():
            print(f"ARN Daemon already running (socket: {SOCKET_PATH})")
            sys.exit(0)
        run_server()

    elif subcmd == "ping":
        r = send_request({"cmd": "ping"}, timeout=3)
        print(json.dumps(r))

    elif subcmd == "stop":
        r = send_request({"cmd": "shutdown"}, timeout=5)
        print(json.dumps(r))

    elif subcmd == "status":
        if is_running():
            r = send_request({"cmd": "ping"})
            print(f"✅ Running — {r}")
        else:
            print("❌ Not running")

    elif subcmd == "recall":
        query = " ".join(sys.argv[2:])
        if is_running():
            r = send_request({"cmd": "recall", "query": query, "top_k": 5})
            print(json.dumps(r, indent=2))
        else:
            print("⚠️  Daemon not running — falling back to direct CLI")
            os.execv("/usr/bin/python3", ["python3",
                f"{ARN_ROOT}/arn_v9/scripts/arn_cli.py", "recall",
                "--query", query, "--top-k", "5"])

    elif subcmd == "stats":
        r = send_request({"cmd": "stats"})
        print(json.dumps(r, indent=2))

    else:
        print(f"Unknown subcommand: {subcmd}")
        print(__doc__)
        sys.exit(1)
