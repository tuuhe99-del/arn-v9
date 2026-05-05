# ARN Human Memory Core

ARN 1.3 adds a simple human-inspired memory layer on top of the existing semantic memory engine.

The goal is not to simulate a brain perfectly. The goal is to give agents the memory types they need to behave more consistently:

- **episode** — something that happened
- **fact** — stable project/user knowledge
- **preference** — what the user likes or wants
- **identity** — who an agent is and how it should behave
- **rule** — instructions that should be injected before the agent acts
- **procedure** — how to do a task
- **error** — a mistake, fix, or lesson from the past
- **task** — current or historical work
- **decision** — choices made by an agent or manager
- **conflict** — contradictory memories that need attention

## Store facts and preferences

```bash
arn store "Mohamed prefers Python" --type preference --scope user --priority high
arn recall "what does the user like to code in?"
```

## Identity memory

```bash
arn identity set developer \
  --name Koda \
  --role "coding, debugging, scripts, tests" \
  --must "inspect files before changing them" \
  --must "run tests before claiming success" \
  --must-not "only explain when asked to fix code"

arn identity show developer
```

Identity memory is pinned into the context packet for that agent.

## Rule memory

```bash
arn rule add developer "Do not claim a code fix is complete unless tests pass" --priority critical
arn rule add researcher "Cite sources when using web information" --priority high
arn rule list developer
```

Rules are stored as high-priority memories. ARN injects global rules plus agent-specific rules.

## Procedural memory

Procedural memory tells the agent how to do tasks.

```bash
arn procedure add release-audit \
  --agent developer \
  --step "Run python compile check" \
  --step "Run pytest" \
  --step "Run CLI smoke tests" \
  --step "Check OpenClaw hooks" \
  --success "All tests pass and documented commands work"

arn procedure recall "how do I audit a release?" --agent developer
```

## Error and lesson memory

```bash
arn error add \
  --agent developer \
  --task "audit ARN release" \
  --mistake "CLI parser referenced a missing handler" \
  --fix "Add the handler and run CLI smoke tests" \
  --lesson "Every documented command needs a smoke test"

arn error list --agent developer --query "release audit"
```

## Context packets

A context packet is the main thing agents should receive before replying.

```bash
arn context \
  --for-agent developer \
  --task "audit ARN release" \
  "what should I do next?"
```

A packet can include:

1. Identity
2. Rules
3. Current task
4. Relevant procedures
5. Past errors and lessons
6. User preferences
7. Project facts

This keeps ARN simple: it remembers typed memories and assembles the right ones before the agent acts.

## Multi-agent memory

Use scopes to separate memory:

```bash
arn store "Project is a public semantic memory tool" --type fact --scope global
arn rule add global "Never claim validation passed unless test output exists" --priority critical
arn identity set developer --name Koda --role "coding and testing"
arn identity set researcher --name Dex --role "research and citations"
```

Each agent gets global memory plus its own agent-scoped identity, rules, procedures, and lessons.

## Cross-Agent Shared Memory

Agents can pass useful task notes to each other without needing a full chat broker.

```bash
arn --agent-id developer share send \
  "Tests passed after the CLI fix." \
  --from-agent developer \
  --to manager \
  --task "ARN release"

arn --agent-id manager share inbox --agent manager --task "ARN release"
```

Shared notes are stored as `memory_type=shared`. ARN writes an outbox copy for the sender and an inbox copy for each recipient, so the note survives restarts and appears in the recipient's context packet.
