# Python API

```python
from arn import ARNPlugin

with ARNPlugin(agent_id="my-agent") as memory:
    memory.store("User prefers Python", importance=0.9)
    hits = memory.recall("what does the user like to code in?", top_k=5)
```

Useful methods:

- `store(content, importance=0.5, tags=None, source="agent", time_context="current")`
- `recall(query, top_k=5, time_filter=None)`
- `get_context_window(query, max_tokens=1000)`
- `maintain()`
- `list_contradictions()`


## Human Memory Core REST API

All endpoints require `X-API-Key` unless `ARN_API_KEY_DISABLED=1` is set for a trusted local/private network.

### Store typed memory

`POST /v1/memory/store` now accepts the normal memory fields plus:

```json
{
  "agent_id": "my-agent",
  "content": "Mohamed prefers Python",
  "memory_type": "preference",
  "scope": "user",
  "priority": "high",
  "time_context": "current"
}
```

### Identity

`POST /v1/human/identity/set`

```json
{
  "agent_id": "manager",
  "target_agent": "developer",
  "name": "Koda",
  "role": "coding, debugging, tests",
  "must": ["run tests before claiming success"],
  "must_not": ["only explain when asked to fix code"]
}
```

`POST /v1/human/identity/get`

```json
{ "agent_id": "manager", "target_agent": "developer" }
```

### Rules

`POST /v1/human/rule/add`

```json
{
  "agent_id": "manager",
  "target_agent": "developer",
  "rule": "Do not claim success unless tests pass",
  "priority": "critical"
}
```

`POST /v1/human/rule/list`

```json
{ "agent_id": "manager", "target_agent": "developer" }
```

### Procedures

`POST /v1/human/procedure/add`

```json
{
  "agent_id": "manager",
  "target_agent": "developer",
  "name": "release-audit",
  "steps": ["compile Python", "run pytest", "run CLI smoke tests"],
  "success": "all tests pass"
}
```

`POST /v1/human/procedure/recall`

```json
{
  "agent_id": "manager",
  "target_agent": "developer",
  "query": "how do I audit a release?",
  "top_k": 5
}
```

### Error lessons

`POST /v1/human/error/add`

```json
{
  "agent_id": "manager",
  "target_agent": "developer",
  "mistake": "CLI handler was missing",
  "fix": "add the handler",
  "lesson": "smoke-test every documented CLI command"
}
```

`POST /v1/human/error/list`

```json
{ "agent_id": "manager", "target_agent": "developer", "query": "CLI audit", "limit": 10 }
```

### Context packet

`POST /v1/human/context-packet`

```json
{
  "agent_id": "manager",
  "target_agent": "developer",
  "task": "audit ARN release",
  "query": "what should I do next?",
  "max_tokens": 2000
}
```

Returns a prompt-ready packet containing identity, rules, current task, relevant procedures, past errors/lessons, preferences, and facts.

## Cross-Agent Sharing Endpoints

ARN 1.4 adds simple cross-agent shared memory endpoints.

### POST `/v1/human/share/send`

Stores an outbox copy in the sender's memory and an inbox copy in each recipient's memory.

```json
{
  "agent_id": "developer",
  "from_agent": "developer",
  "to_agents": ["manager", "researcher"],
  "content": "Tests passed after the CLI fix.",
  "task": "ARN release",
  "importance": 0.75,
  "priority": "high",
  "tags": ["release"]
}
```

### POST `/v1/human/share/list`

Lists all shared memories visible to an agent.

```json
{
  "agent_id": "manager",
  "target_agent": "manager",
  "task": "ARN release",
  "limit": 20
}
```

### POST `/v1/human/share/inbox`

Lists only received shared memories.

### POST `/v1/human/share/outbox`

Lists only sent shared memories.
