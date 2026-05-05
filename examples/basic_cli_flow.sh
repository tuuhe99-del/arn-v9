#!/usr/bin/env bash
set -euo pipefail

export OPENCLAW_AGENT_ID="demo-agent"
export ARN_DATA_ROOT="./demo_memory"
export ARN_EMBEDDING_TIER="nano"

arn-cli check || true
arn-cli store --content "User's DCO website runs locally on port 4173" --importance 0.8 --tags "project,dco,port" --time-context current
arn-cli recall --query "What port does the DCO website use?" --top-k 3
arn-cli context --query "DCO website troubleshooting" --max-tokens 500
