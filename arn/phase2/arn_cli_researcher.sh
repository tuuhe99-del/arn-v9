#!/bin/bash
export OPENCLAW_AGENT_ID="researcher"
export PYTHONPATH="/home/mokali/arn_v9:$PYTHONPATH"
exec python3 /home/mokali/arn_v9/arn_v9/scripts/arn_cli.py "$@"
