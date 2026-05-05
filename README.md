# ARN — Semantic Memory for AI Agents

ARN is a local semantic memory layer for AI agents.

It lets an agent store facts, remember them across sessions, and recall the relevant context by **meaning**, not exact keyword matching.

```bash
arn store "Mohamed prefers Python"
arn recall "what does the user like to code in?"
```

Expected result:

```text
Mohamed prefers Python
```

The query does not contain the words `prefers` or `Python`, but ARN retrieves the memory because the meaning matches.

## Why use ARN?

Most agent memory systems either store raw chat logs or search by keywords. ARN is built for agent context:

- **Semantic recall** — sentence embeddings retrieve memories by meaning.
- **Persistent memory** — memories survive restarts and new sessions.
- **Episodic + semantic memory** — recent events are stored as episodes; repeated patterns can consolidate into stable facts.
- **Contradiction detection** — conflicting facts are flagged instead of silently overwritten.
- **Temporal tagging** — memories can be marked as `past`, `current`, `future`, or `timeless`.
- **Model profiles** — switch between lightweight and stronger embedding models based on hardware.
- **Agent integrations** — CLI, Python API, OpenClaw hook, and generic hook examples.

## Install

### Linux / macOS

```bash
bash install.sh --tier nano
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -tier nano
```

### Universal Python installer

```bash
python install.py --tier nano
```

The installer creates a local virtual environment, installs dependencies, downloads/verifies the selected embedding model, and creates the `arn` command.

## Quick start

```bash
arn doctor
arn selftest --strict --isolated

arn store "Mohamed prefers Python" --importance 0.9 --tags preference,coding
arn recall "what does the user like to code in?"
arn context "coding preferences"
```

## CLI commands

```bash
arn store "User prefers beginner-friendly explanations"
arn recall "how should I explain this to the user?"
arn list
arn search "Python"                  # keyword search fallback
arn contradictions
arn maintain                          # consolidation/maintenance
arn doctor                            # diagnose install/model issues
arn models list
arn models recommend
arn models switch --tier base --download
```

## Python usage

```python
from arn import ARNPlugin

with ARNPlugin(agent_id="demo") as memory:
    memory.store("Mohamed prefers Python", importance=0.9, tags=["preference"])
    results = memory.recall("what does the user like to code in?", top_k=3)
    print(results[0]["content"])
```

## OpenClaw integration

ARN includes a first-class OpenClaw hook that can automatically:

1. store inbound user messages,
2. inject relevant memory before the agent replies,
3. store useful assistant replies after the response is sent.

See [`docs/OPENCLAW_INTEGRATION.md`](docs/OPENCLAW_INTEGRATION.md).

## Embedding model tiers

| Tier | Good for | Approx RAM | Notes |
|---|---:|---:|---|
| `nano` | Raspberry Pi / low RAM | 120 MB | default; MiniLM |
| `small` | Pi 5 with spare RAM | 250 MB | BGE small |
| `balanced` | laptop/desktop | 550 MB | MPNet |
| `base` | stronger recall | 650 MB | BGE base |
| `base-e5` | retrieval-heavy agents | 650 MB | E5 base |
| `large` | 16GB+ machines | 1.4 GB | BGE large |

See [`docs/EMBEDDING_MODELS.md`](docs/EMBEDDING_MODELS.md).

## Privacy

ARN runs locally. Your memory data is stored under `~/.arn_data` by default. Embedding models are downloaded to your machine, but memory content is not sent to an ARN cloud service.

## Project layout

```text
arn/                    # core Python package
  core/                 # embeddings, cognitive memory, contradictions
  storage/              # local persistence
  api/                  # optional FastAPI server/UI
  scripts/              # CLI entrypoint
integrations/           # OpenClaw + generic hook examples
openclaw/               # installable OpenClaw hook files
examples/               # beginner demos
docs/                   # install, behavior, troubleshooting
tests/                  # source tests
benchmarks/             # optional development benchmarks
```

## License

MIT. See [`LICENSE`](LICENSE).

## Human Memory Core (ARN 1.3)

ARN now supports simple human-inspired memory types so agents can remember more than facts:

```bash
arn store "Mohamed prefers Python" --type preference --scope user --priority high
arn identity set developer --name Koda --role "coding, debugging, tests" --must "run tests before claiming success"
arn rule add developer "Do not claim success unless tests pass" --priority critical
arn procedure add release-audit --step "Compile Python" --step "Run pytest" --step "Run CLI smoke tests" --agent developer
arn error add --agent developer --mistake "CLI handler was missing" --fix "Add handler" --lesson "Smoke test documented commands"
arn context --for-agent developer --task "audit ARN release" "what should I do next?"
```

The context packet pulls identity, rules, current task, procedures, past errors, preferences, and facts into one prompt-ready block. See `docs/HUMAN_MEMORY_CORE.md`.



## Cross-Agent Communication

Agents can share useful task memories with each other:

```bash
arn --agent-id developer share send "Tests passed after the CLI fix" --from-agent developer --to manager --task "ARN release"
arn --agent-id manager share inbox --agent manager --task "ARN release"
```

Shared notes are included automatically in relevant context packets. See `docs/CROSS_AGENT_COMMUNICATION.md`.
