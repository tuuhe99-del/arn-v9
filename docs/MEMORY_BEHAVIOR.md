# ARN Memory Behavior

ARN's core promise is simple:

```bash
arn store "Mohamed prefers Python"
arn recall "what does the user like to code in?"
```

A healthy install returns the Python memory because recall is semantic, not keyword-only.

## Memory loop

1. **Perceive/store** — incoming facts, messages, decisions, and tool results are stored as episodes.
2. **Embed** — the text is converted into a vector using the selected embedding model.
3. **Recall** — queries are embedded and compared with stored memory vectors.
4. **Inject** — integrations can place relevant memories into the agent context before a reply.
5. **Consolidate** — repeated or important episodes can become longer-lived semantic memories.
6. **Protect history** — contradictions are flagged and both memories are retained with timestamps.

## Episodic vs semantic memory

- **Episodic memory** stores concrete events: messages, actions, observations, and tool results.
- **Semantic memory** stores consolidated knowledge: stable preferences, recurring facts, and patterns.

## Contradictions

ARN does not erase old claims when new information conflicts. It keeps both and attaches conflict metadata.

Example:

```bash
arn store "Mohamed prefers Python" --tags preference
arn store "Mohamed prefers JavaScript" --tags preference
arn contradictions
```

## Temporal tags

Use `--time-context` when a fact has a clear time meaning:

```bash
arn store "Mohamed used to prefer JavaScript" --time-context past
arn store "Mohamed currently prefers Python" --time-context current
arn store "Mohamed plans to learn Rust" --time-context future
```
