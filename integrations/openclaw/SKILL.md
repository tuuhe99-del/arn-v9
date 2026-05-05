---
name: arn-memory
description: >
  Persistent semantic memory for OpenClaw-style agents. Use it to remember
  durable facts, recall relevant context, inject memory before replies, and
  store important messages or tool results after they happen.
metadata:
  openclaw:
    requires:
      bins:
        - arn
    os:
      - linux
    emoji: "🧠"
    always: true
---

# ARN Memory Skill

This skill uses the installed `arn` command. Do not hard-code a source path.
Set `OPENCLAW_AGENT_ID` to the current agent name so each agent gets its own memory namespace.

## Automatic OpenClaw hook integration

The installer places a managed OpenClaw hook here:

```bash
~/.openclaw/hooks/arn-memory/
```

Enable it:

```bash
openclaw hooks enable arn-memory
openclaw hooks check
```

The hook connects ARN to OpenClaw's lifecycle:

```text
message:received     → store inbound message
message:preprocessed → recall memory and prepend it to bodyForAgent
agent:bootstrap      → inject ARN_MEMORY.md bootstrap context when available
message:sent         → store outbound assistant reply
session:compact:before → store a compaction marker
```

## Manual commands

Use these when the automatic hook is unavailable or when building your own CLI/Telegram bridge.

```bash
# When a message arrives
arn --strict hook receive --role user --message "<incoming message>"

# Before replying; inject this output into the prompt
arn --strict hook before-reply --query "<current task>" --max-tokens 1000

# One-step preprocessing: store inbound + print memory-prefixed message
arn --strict hook preprocessed --message "<incoming message>" --max-tokens 1000

# After an important answer is sent
arn --strict hook send --role assistant --message "<summary of useful answer>"

# After a tool/command returns useful output
arn --strict hook tool-result --tool-name "<tool>" --message "<important output>"
```

## When to store

Store information whenever you learn something worth remembering:

1. User identity or role facts → importance 0.9
2. User preference → importance 0.7-0.8
3. Project decision or constraint → importance 0.8
4. Bug found and fixed → importance 0.7
5. Technical approach chosen → importance 0.6
6. Routine chatter → importance 0.1-0.2

Never store passwords, API keys, tokens, or credentials.

## Time context

Use `--time-context` correctly:

- `current` — facts true right now
- `past` — old facts that used to be true
- `future` — plans or intentions

Examples:

```bash
arn --strict store \
  --content "User prefers Python for scripting" \
  --importance 0.8 \
  --time-context current

arn --strict store \
  --content "User used to run the site on port 4173" \
  --importance 0.7 \
  --time-context past

arn --strict store \
  --content "User plans to migrate the dashboard next week" \
  --importance 0.8 \
  --time-context future
```

## Troubleshooting

```bash
arn --strict check
arn doctor
```

If strict check fails, semantic embeddings did not load. Do not trust recall until the installer problem is fixed.
