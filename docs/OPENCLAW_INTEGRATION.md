# OpenClaw Integration

ARN supports OpenClaw as an optional integration. ARN itself is a general semantic memory tool; OpenClaw is one supported agent runtime.

## Intended flow

```text
message:received
  -> ARN stores inbound user message

message:preprocessed
  -> ARN recalls relevant memories
  -> hook prepends memory context to bodyForAgent when supported

agent:bootstrap
  -> ARN can inject an ARN_MEMORY.md bootstrap file when supported

message:sent
  -> ARN stores useful assistant replies/decisions
```

## Install

```bash
bash install.sh --tier nano
```

The installer copies files into:

```text
~/.openclaw/skills/arn-memory/SKILL.md
~/.openclaw/hooks/arn-memory/HOOK.md
~/.openclaw/hooks/arn-memory/handler.ts
```

If the OpenClaw CLI is available, the installer attempts to enable the hook automatically. If not, enable it manually later:

```bash
openclaw hooks list
openclaw hooks enable arn-memory
```

## Manual hook testing

```bash
arn hook receive --role user --message "Mohamed prefers Python"
arn hook before-reply --query "what does the user like to code in?"
arn hook send --role assistant --message "I remembered the Python preference."
```

## Safety behavior

The OpenClaw hook should never break message delivery. If ARN fails, the hook logs the error and lets OpenClaw continue.

Set a custom CLI path if needed:

```bash
export ARN_CLI=/full/path/to/arn
```
