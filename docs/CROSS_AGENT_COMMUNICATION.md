# Cross-Agent Communication

ARN supports simple cross-agent memory sharing. It is intentionally not a message broker, queue system, or chat server. The goal is smaller and more useful: agents can pass important task knowledge to other agents, and those notes become part of the recipient agent's normal memory/context.

## Mental model

When one agent shares a memory, ARN stores:

1. an **outbox** copy in the sender's memory, and
2. an **inbox** copy in each recipient's memory.

This keeps the system local and reliable. Each agent can recall shared information using the same semantic memory and context packet commands it already uses.

## CLI

Share a task note from the developer to the manager and researcher:

```bash
arn --agent-id developer share send \
  "The CLI store bug was fixed and pytest passes." \
  --from-agent developer \
  --to manager \
  --to researcher \
  --task "ARN 1.4 release"
```

Read the manager inbox:

```bash
arn --agent-id manager share inbox --agent manager --task "ARN 1.4 release"
```

Read the developer outbox:

```bash
arn --agent-id developer share outbox --agent developer
```

List all shared memories visible to an agent:

```bash
arn --agent-id researcher share list --agent researcher --query "CLI bug"
```

## Context packets

Shared notes are automatically included in context packets when relevant:

```bash
arn --agent-id manager context --for-agent manager --task "ARN 1.4 release" "what happened?"
```

The packet can include:

```text
## Shared Agent Notes
- Shared memory from agent 'developer'.
  Task: ARN 1.4 release
  Content: The CLI store bug was fixed and pytest passes.
```

## REST API

Send a shared note:

```http
POST /v1/human/share/send
```

```json
{
  "agent_id": "developer",
  "from_agent": "developer",
  "to_agents": ["manager", "researcher"],
  "content": "The CLI store bug was fixed and pytest passes.",
  "task": "ARN 1.4 release"
}
```

List shared memories:

```http
POST /v1/human/share/list
POST /v1/human/share/inbox
POST /v1/human/share/outbox
```

```json
{
  "agent_id": "manager",
  "target_agent": "manager",
  "task": "ARN 1.4 release",
  "limit": 20
}
```

## Why this is simple

ARN does not require every agent to be online at the same time. Sharing writes a durable note into the recipient's memory folder. The next time that agent starts, the shared memory is already there.
