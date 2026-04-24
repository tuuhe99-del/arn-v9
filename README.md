# ARN — Adaptive Reasoning Network

**Persistent memory for AI agents. Local. Free. Works on a Raspberry Pi.**

Your AI agent forgets everything between sessions. ARN fixes that — it remembers facts, preferences, decisions, and past conversations across restarts, without sending anything to the cloud.

---

## Install in two commands

```bash
curl -fsSL https://raw.githubusercontent.com/tuuhe99-del/arn-v9/main/install.sh | bash
```

Then connect it to your agent:

```bash
arn connect
```

`arn connect` detects your hardware, picks the right model size, and wires everything up automatically. That's it — your agent now has memory.

---

## What it does

```
Session 1:
  You: "My name is Mohamed and I prefer Python"
  Agent: stores this automatically

Session 2 (days later, after restart):
  You: "What's my name?"
  Agent: "Mohamed" ← pulled from memory, not re-explained
```

ARN stores facts as semantic vectors — meaning it finds relevant memories by *meaning*, not keyword match. Ask "what language does the user like to code in?" after storing "prefers Python" and it finds it, even though none of the words match exactly.

---

## Why ARN over mem0, Zep, or LangMem?

| | **ARN** | mem0 | Zep | LangMem |
|---|---|---|---|---|
| Cost | **$0/month** | $0–$249/mo | $0–$25+/mo | $0 |
| Runs locally | ✅ fully offline | ⚠️ complex self-host | ❌ cloud only | ✅ library |
| Works on Raspberry Pi | ✅ tested | ❌ too heavy | ❌ cloud | ⚠️ |
| Needs LLM API for memory | **No** | Yes (GPT default) | Yes | Yes |
| Temporal tagging (past vs now) | ✅ built-in | ❌ | ⚠️ partial | ❌ |
| Auto-inject (passive absorption) | ✅ | ❌ manual | ❌ manual | ⚠️ |
| Setup | **2 commands** | API keys + config | Cloud signup | LangGraph required |
| Framework lock-in | None | None | None | LangChain only |

**Where competitors win:** mem0 has enterprise compliance (SOC 2, HIPAA) and knowledge graphs. Zep has entity extraction. LangMem has procedural memory for LangGraph agents. If you need those, use them.

**Where ARN wins:** you want memory that runs on your own hardware, costs nothing per month, and doesn't require an API key just to store a fact.

---

## How it actually works (the short version)

Most memory systems do: store embedding → vector search → inject top results. That breaks in common situations.

**ARN does it differently:**

**1. Temporal tagging** — "User used to prefer React, now prefers Vue." A plain vector search returns both with similar scores. ARN tags every memory as `past`, `current`, or `future` and filters at recall time so you always get the right version.

**2. Cortical column voting** — 8 domain-specialized processors (code, conversation, facts, errors, decisions, etc.) each score incoming memories independently. Important information from any domain gets stored correctly. Noise gets filtered.

**3. Episodic → semantic consolidation** — Like how human sleep consolidates memories. Recent facts stay as specific episodes. Repeated patterns get consolidated into general knowledge. This keeps storage efficient and recall relevant long-term.

**4. Prediction error weighting** — Surprising information gets stored with higher priority. If you already know something, ARN won't re-store it with the same weight. Novel facts stand out.

**5. Auto-inject (Phase 2)** — Passive memory layer. ARN automatically absorbs every user message and injects relevant context at the start of each session. The agent just knows — no tool calls, no "let me check my memory."

---

## Model tiers — pick based on your hardware

The installer picks automatically. You can override:

```bash
bash install.sh --model nano    # 22MB  — Raspberry Pi, anything with <1GB free RAM
bash install.sh --model small   # 420MB — laptops, mid-range hardware
bash install.sh --model base    # 440MB — recommended for most desktops ← default on 6-12GB RAM
bash install.sh --model large   # 1.3GB — high-end desktop or server
bash install.sh --model xl      # ~14GB — GPU server, top-tier quality
```

| Tier | Model | Size | RAM | Best for |
|------|-------|------|-----|----------|
| `nano` | all-MiniLM-L6-v2 | 22MB | ~90MB | Raspberry Pi, low-power devices |
| `small` | all-mpnet-base-v2 | 420MB | ~420MB | Laptops, Pi 5 with 8GB RAM |
| `base` | BAAI/bge-base-en-v1.5 | 440MB | ~500MB | Most desktops — good balance |
| `large` | BAAI/bge-large-en-v1.5 | 1.3GB | ~1.3GB | High-end desktop / server |
| `xl` | intfloat/e5-mistral-7b-instruct | ~14GB | 2.5GB+ | GPU machine, maximum quality |

Switch tier any time — just set the env var and restart the daemon:
```bash
export ARN_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
arn daemon stop && arn daemon start
```

---

## Usage — for everyone

### Command line (simplest)

```bash
# Store something
arn store "My name is Mohamed and I'm studying IT at Columbus State"
arn store "I prefer Python over JavaScript" --importance 0.8
arn store "I used to work at the coffee shop on campus" --time-context past

# Recall by meaning — not keyword
arn recall "what does the user study?"
arn recall "programming language preference"
arn recall "where did user work before?"

# Get a full context block (ready to paste into a system prompt)
arn context "user background and preferences"

# Check what's stored
arn stats

# Check daemon status
arn ping
```

### Python — basic

```python
from arn import ARNv9

arn = ARNv9(data_dir="~/.arn_data/default")

# Store facts
arn.perceive("User prefers Python for scripting", importance=0.8)
arn.perceive("Deployed on Raspberry Pi 5 with 8GB RAM", importance=0.7)

# Recall by meaning
results = arn.recall("what language does the user prefer?", top_k=3)
for r in results:
    print(f"[{r['score']:.2f}] {r['content']}")
# Output:
# [0.82] User prefers Python for scripting

arn.close()
```

### Python — with temporal tagging (recommended)

```python
from arn import ARNPlugin

with ARNPlugin(agent_id="my_agent", data_root="~/.arn_data") as arn:

    # Tag facts with WHEN they're true
    arn.store("User prefers Vue for frontend", importance=0.8, time_context="current")
    arn.store("User used to prefer React before 2024", importance=0.6, time_context="past")
    arn.store("User plans to learn Rust next month", importance=0.5, time_context="future")

    # Recall — temporal filter applied automatically
    hits = arn.recall("frontend framework preference", top_k=3)
    for h in hits:
        print(f"[{h['score']:.2f}] [{h['time_context']}] {h['content']}")
    # Output:
    # [0.81] [current] User prefers Vue for frontend
    # [0.54] [past] User used to prefer React before 2024
```

### Python — auto-inject (Phase 2, fully passive)

```python
from arn import AutoInject

# Works with any OpenAI-compatible LLM (Ollama, LM Studio, etc.)
ai = AutoInject(
    agent_id="my_agent",
    backend="ollama",
    model="qwen2.5:0.5b",   # or any model you have
)

# Memory is automatic — no store/recall calls needed
response = ai.chat("Hi, my name is Mohamed and I prefer Python")
# ARN stores this in the background

response = ai.chat("What's my name?")
print(response)  # → "Mohamed"
# Next session after restart — still knows your name
```

### Drop-in adapter — any framework

```python
from arn.adapters.raw import ARNMemory

with ARNMemory() as mem:
    # Store
    mem.store("User prefers dark mode", importance=0.7)
    mem.store("User is building a Telegram bot", importance=0.8)

    # Recall
    hits = mem.recall("what is user building?", top_k=3)
    print(hits[0]["content"])  # → User is building a Telegram bot

    # Get formatted context block for system prompt injection
    context = mem.context("user preferences and projects")
    print(context)
    # → [MEMORY - current] User prefers dark mode
    #   [MEMORY - current] User is building a Telegram bot
```

---

## Connecting to your agent framework

### OpenClaw

`arn connect` handles this automatically. The memory skill gets installed to all your agents.

Manual install:
```bash
mkdir -p ~/.openclaw/workspace/agents/<agent-name>/skills/arn-memory
cp arn/openclaw_skill/SKILL.md ~/.openclaw/workspace/agents/<agent-name>/skills/arn-memory/
```

Once installed, your agent can use:
```bash
arn store "fact" --importance 0.8
arn recall "query"
arn context "topic"
```

### LangChain

```python
from arn.adapters.langchain import get_tools
from langchain.agents import initialize_agent

# Get ARN as LangChain tools
tools = get_tools(data_dir="~/.arn_data/default")
# Returns: remember() and recall_memory() tools

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# Agent now has persistent memory automatically
agent.run("Remember that I prefer dark mode")
agent.run("What are my preferences?")  # ← recalls from ARN
```

### Claude (system prompt method)

Add this to your Claude system prompt or AGENTS.md:

```
You have persistent memory via ARN.

To store a fact worth remembering:
  arn store "<fact>" --importance 0.8

To recall relevant context:
  arn recall "<question>"

To get full context for a topic:
  arn context "<topic>"

Rules:
- Store when: user shares preferences, makes decisions, tells you something personal
- Recall before answering anything where past context might matter
- Use --time-context past when storing something that used to be true
```

### Custom / any other framework

```python
from arn.adapters.raw import ARNMemory

mem = ARNMemory(data_dir="~/.arn_data/default")

# In your pre-processing step:
context = mem.context("relevant topic for this message")
system_prompt = base_system_prompt + "\n\nWhat you remember:\n" + context

# In your post-processing step:
mem.store(user_message, importance=0.5)  # absorb user messages passively
```

---

## Daemon (keeps recall fast)

Without the daemon, each recall call loads the embedding model from scratch (~10s cold start). The daemon keeps it loaded in RAM — 0.5s recall instead.

```bash
arn daemon start    # start in background
arn daemon stop     # stop
arn daemon status   # check if running
arn ping            # quick ping
```

The installer sets up the daemon automatically via systemd (if available) or background process.

---

## Stress test results

These aren't cherry-picked — they're from `tests/stress_test.py`, which you can run yourself:

| Test | Result |
|------|--------|
| Cross-session persistence (4 restarts + noise injection) | ✅ 3/3 facts recalled |
| Distractor resistance (5 target facts in 500 noise facts) | ✅ 5/5 found |
| Contradiction handling | ✅ most-recent version always wins |
| Temporal reasoning (past vs current filtering) | ✅ current always beats past |
| Hallucination refusal (low-confidence detection) | ✅ 3/3 flagged correctly |
| Paraphrase robustness (reworded queries) | ✅ 6/6 hit |
| Scale — 1,000 episodes | ✅ 100% accuracy, ~170ms |
| Scale — 3,000 episodes | ✅ 100% accuracy, ~170ms |

Run them yourself:
```bash
cd ~/arn
python3 -m pytest tests/ -v
```

---

## Troubleshooting

**"command not found: arn"** — restart your shell or run `source ~/.bashrc`

**"No module named arn"** — run `pip install -e ~/arn`

**Recall takes 10+ seconds** — daemon isn't running: `arn daemon start`

**Daemon won't start** — first run downloads the embedding model (22MB–1.3GB depending on tier). Check progress: `tail -f ~/.arn_daemon.log`

**Nothing gets recalled** — run `arn stats` to confirm episodes were stored. Check `ARN_DATA_DIR` is the same path used when storing.

**On Raspberry Pi, running out of RAM** — force nano tier:
```bash
export ARN_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
arn daemon stop && arn daemon start
```

**Want to wipe memory and start fresh:**
```bash
rm -rf ~/.arn_data/default
```

---

## File structure

```
arn/
├── core/           # embedding engine + cognitive architecture
│   ├── cognitive.py    # cortical columns, episodic/semantic split, consolidation
│   ├── embeddings.py   # model registry, all 5 tiers
│   └── __init__.py
├── storage/        # SQLite + memmap persistence
├── phase2/         # auto-inject layer (passive absorption)
│   └── memory_llm.py   # MemoryAugmentedLLM / AutoInject
├── adapters/       # framework integrations
│   ├── openclaw.py
│   ├── langchain.py
│   └── raw.py          # ARNMemory — works with anything
├── plugin.py       # ARNPlugin — temporal tagging + full API
├── cli.py          # arn command entrypoint
└── __init__.py     # exports: ARNv9, ARNPlugin, AutoInject

install.sh          # one-line installer
tests/              # full test suite
```

---

## License

**Free** for personal use, students, researchers, and open-source projects.

**Paid license required** for companies over $1M annual revenue using this in a product.

See [LICENSE.md](./LICENSE.md) and [COMMERCIAL.md](./COMMERCIAL.md). The short version: if you're a student or indie dev, use it for free. If a company is making money from it, reach out.

---

Built by Mohamed Mohamed — IT student at Columbus State Community College, on a Raspberry Pi 5.

[GitHub](https://github.com/tuuhe99-del/arn-v9) · [Issues](https://github.com/tuuhe99-del/arn-v9/issues)
