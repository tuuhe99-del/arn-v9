# Changelog

## 1.4.0 - Cross-Agent Communication

- Added simple cross-agent shared memory.
- Added `arn share send`, `arn share inbox`, `arn share outbox`, and `arn share list`.
- Added REST endpoints under `/v1/human/share/*`.
- Context packets now include relevant shared agent notes.
- Shared notes are duplicated into recipient agent namespaces to keep the design local, simple, and reliable.

## 1.3.1 - API/Web UI Human Memory Patch

- Added REST endpoints for human memory core features: identity, rules, procedures, error lessons, and context packets.
- Updated `/v1/memory/store` to accept `memory_type`, `scope`, `priority`, `time_context`, and `name`.
- Added a simple Web UI Human tab for identity/rule/procedure/error/context-packet workflows.
- Removed release cache artifacts such as `.pytest_cache/` from packaged archives.
- Added API tests for human memory endpoint round trips.

## 1.3.1 - Human Memory Core

- Added typed memory metadata: episode, fact, preference, identity, rule, procedure, error, lesson, task, decision, conflict.
- Added agent identity commands: `arn identity set/show`.
- Added rule memory commands: `arn rule add/list`.
- Added procedural memory commands: `arn procedure add/recall/list`.
- Added error/lesson memory commands: `arn error add/list`.
- Added context packets with identity, rules, task, procedures, past lessons, preferences, and facts.
- Kept design simple by reusing existing ARN storage and embeddings instead of adding another database.


## 1.2.1

### Release-blocking fixes
- Restored the missing `cmd_store` CLI handler so `arn store`, `arn recall`, and every CLI command can start correctly.
- Hardened default embedding-tier parsing so invalid external argv/env values do not crash imports.
- Prevented benchmark scripts from being collected/executed during normal pytest runs.
- Added `pytest.ini` so CI runs only the intended test suite by default.

## 1.2.0

### New features
- **Auto-extraction** — `store()` now automatically detects facts from free text at write time. Preferences, locations, roles, tool usage, port numbers, software versions, goals, and ownership are all extracted without the agent explicitly categorising them. Adds matching tags to context automatically.
- **Temporal supersession** — when a new memory contradicts an older one on the same fact (e.g. "User switched from Python to Rust"), the old episode is marked `superseded_by` the new one. Recall applies a penalty to superseded episodes so the current fact surfaces first. Full history is preserved.
- **Fact graph** — a lightweight entity→relation→object graph (networkx, no server required) is built automatically from extracted claims. Persisted as JSON in the agent's existing SQLite database — no separate file. New CLI commands: `arn graph query <entity>`, `arn graph history <entity> <relation>`, `arn graph summary`. New API endpoints: `GET /v1/memory/graph/{agent_id}`.
- **Change detection** — `store()` detects language like "switched to", "no longer", "now prefers" and boosts importance of the new fact so it ranks above older versions in recall.
- **DB migration** — existing databases from 1.0.x and 1.1.x are migrated automatically on first open. No manual steps, no data loss.

### Bug fixes
- Fixed `test_full_integration` missing `@requires_embeddings` decorator — caused 2 false test failures in degraded mode.
- Extended extraction patterns: location, role, goal, port, version now detected alongside existing preference/uses/identity patterns.

### Dependencies
- Added `networkx>=3.0` (lightweight graph library, no server needed).

## 1.1.0

- Reframed ARN as a public semantic memory tool for any agent.
- Kept OpenClaw as an optional first-class integration.
- Added `arn-cli` command alias alongside `arn`.
- Added beginner positional commands: `arn store "..."` and `arn recall "..."`.
- Added public GitHub-ready docs, license, contributing guide, and CI workflow.
- Moved tests and benchmarks out of the installed `arn` package.
- Removed default phone-home update URL; update checks only run when `ARN_UPDATE_URL` is set.
- Hardened OpenClaw hook CLI resolution to try `arn`, `arn-cli`, and `ARN_CLI`.

## 1.0.0

- Initial ARN package with semantic recall, model tiers, contradiction tracking, temporal tags, CLI, and OpenClaw hooks.
