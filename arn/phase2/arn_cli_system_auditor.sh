#!/bin/bash
export OPENCLAW_AGENT_ID="system-auditor"
export PYTHONPATH="/home/mokali/arn_v9:$PYTHONPATH"
exec python3 /home/mokali/arn_v9/arn/phase2/arn_cli.py "$@"
