---
name: arn-memory
description: "Automatically inject ARN memory into OpenClaw conversations and store inbound/outbound messages."
metadata:
  openclaw:
    emoji: "🧠"
    events:
      - message:received
      - message:preprocessed
      - message:sent
      - agent:bootstrap
      - session:compact:before
      - gateway:startup
    requires:
      bins:
        - arn
    os:
      - linux
      - macos
      - windows
    always: true
---

# ARN Memory Hook

This hook connects OpenClaw's message lifecycle to ARN.

What it does:

1. `message:received` stores inbound user/channel messages.
2. `message:preprocessed` retrieves relevant ARN memory and prepends it to `bodyForAgent` when OpenClaw exposes that field as mutable.
3. `agent:bootstrap` injects a lightweight ARN memory bootstrap file when OpenClaw exposes `context.bootstrapFiles`.
4. `message:sent` stores useful outbound assistant replies after delivery.
5. `session:compact:before` stores a small compaction marker so ARN knows the session was compressed.

Environment variables:

- `ARN_CLI` — path to the ARN CLI. Default: `arn`.
- `OPENCLAW_AGENT_ID` — memory namespace. Defaults to OpenClaw agent id when available, otherwise `default`.
- `ARN_DATA_ROOT` — memory data directory. Default: `~/.arn_data`.
- `ARN_EMBEDDING_TIER` — optional override. Supported: `nano`, `small`, `balanced`, `base`, `base-e5`, `large`. If unset, ARN reads `~/.arn_config.json`.
- `ARN_HOOK_LOG` — hook log path. Default: `~/.arn_data/openclaw-hook.log`.

Enable:

```bash
openclaw hooks enable arn-memory
openclaw hooks check
```

Switch models without editing the hook:

```bash
arn models switch --tier nano --download
# restart OpenClaw after switching
```
