# Embedding Models

ARN can switch embedding tiers depending on hardware.

```bash
arn models list
arn models recommend
arn models download --tier nano
arn models switch --tier base --download
```

## Tiers

| Tier | Model | Dimension | Best for |
|---|---|---:|---|
| `nano` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Raspberry Pi, low RAM, default |
| `small` | `BAAI/bge-small-en-v1.5` | 384 | better low-RAM recall |
| `balanced` | `sentence-transformers/all-mpnet-base-v2` | 768 | general laptop/desktop use |
| `base` | `BAAI/bge-base-en-v1.5` | 768 | stronger recall quality |
| `base-e5` | `intfloat/e5-base-v2` | 768 | retrieval-heavy agents |
| `large` | `BAAI/bge-large-en-v1.5` | 1024 | strong machines / servers |

## Safe switching

ARN stores each model tier under a separate data folder so mismatched vector dimensions do not corrupt memory.

To re-embed memories into a new tier:

```bash
arn models migrate --from-tier nano --to-tier base --download --consolidate
```
