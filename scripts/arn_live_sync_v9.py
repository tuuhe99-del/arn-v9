#!/usr/bin/env python3
"""
arn_live_sync_v9.py — Live ARN memory sync using arn_v9 engine

Polls memory.db for new agent_knowledge rows and immediately digests them
into arn_v9 via the perceive command. Replaces old arn_simulation live sync.

State file: /home/mokali/.arn_data/default/sync_state.json
"""
from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
import subprocess
from pathlib import Path

MEMORY_DB     = Path("/home/mokali/.openclaw/shared/memory.db")
STATE_FILE    = Path("/home/mokali/.arn_data/default/sync_state.json")
ARN_CLI       = Path("/home/mokali/arn_v9/arn_v9/scripts/arn_cli.py")
POLL_INTERVAL = 16   # seconds — aligned with event bus tick

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [arn-live-sync-v9] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("arn_live_sync_v9")

TRACKED_AGENTS = {
    "manager", "developer", "jr-dev", "researcher",
    "system-analyst", "file-manager", "backupdev",
}


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{MEMORY_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_new_rows(conn: sqlite3.Connection, agent: str, last_id: int) -> list:
    cur = conn.execute(
        "SELECT id, agent, content, created_at FROM agent_knowledge "
        "WHERE agent = ? AND id > ? ORDER BY id LIMIT 50",
        (agent, last_id),
    )
    return cur.fetchall()


def perceive(content: str, source: str) -> bool:
    """Send content to arn_v9 via CLI."""
    try:
        result = subprocess.run(
            [sys.executable, str(ARN_CLI), "perceive",
             "--content", content[:800],
             "--source", source,
             "--min-importance", "0.1"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return True
        log.warning(f"perceive failed: {result.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        log.warning("perceive timed out")
        return False
    except Exception as e:
        log.error(f"perceive error: {e}")
        return False


def run_once(state: dict) -> dict:
    try:
        conn = db_connect()
    except Exception as e:
        log.error(f"Cannot connect to memory.db: {e}")
        return state

    total_digested = 0
    with conn:
        for agent in TRACKED_AGENTS:
            last_id = state.get(agent, 0)
            rows = get_new_rows(conn, agent, last_id)
            for row in rows:
                content = row["content"] or ""
                if len(content.strip()) < 10:
                    state[agent] = row["id"]
                    continue
                ok = perceive(content, agent)
                if ok:
                    total_digested += 1
                state[agent] = row["id"]

    if total_digested > 0:
        log.info(f"Digested {total_digested} new memories into arn_v9")
        save_state(state)
    return state


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--watch"
    log.info(f"ARN v9 Live Sync starting (mode={mode})")

    state = load_state()

    if mode == "--init":
        # Bootstrap: set last_id to current max so we don't re-process old rows
        try:
            conn = db_connect()
            for agent in TRACKED_AGENTS:
                cur = conn.execute(
                    "SELECT MAX(id) FROM agent_knowledge WHERE agent = ?", (agent,)
                )
                row = cur.fetchone()
                max_id = row[0] or 0
                state[agent] = max_id
                log.info(f"  {agent}: bootstrapped to id={max_id}")
            conn.close()
            save_state(state)
            log.info("Init complete.")
        except Exception as e:
            log.error(f"Init failed: {e}")
        return

    if mode == "--once":
        state = run_once(state)
        save_state(state)
        return

    # --watch (daemon mode)
    log.info(f"Polling every {POLL_INTERVAL}s. State file: {STATE_FILE}")
    while True:
        try:
            state = run_once(state)
        except Exception as e:
            log.error(f"Sync cycle error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
