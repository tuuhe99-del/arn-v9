#!/usr/bin/env bash
# Source this file from a shell-based OpenClaw bridge, Telegram bot bridge,
# or any local agent wrapper.
#
# Example:
#   source ./hooks/arn-openclaw-hooks.sh
#   arn_on_message_received "user" "User says the site runs on port 4173"
#   MEMORY_BLOCK="$(arn_before_reply "Cloudflare tunnel setup")"
#   arn_on_message_sent "assistant" "Explained the tunnel fix"

: "${OPENCLAW_AGENT_ID:=default}"
: "${ARN_DATA_ROOT:=$HOME/.arn_data}"
: "${ARN_EMBEDDING_TIER:=nano}"
: "${ARN_CLI:=arn-cli}"

arn_cli() {
  "$ARN_CLI" --strict \
    --agent-id "$OPENCLAW_AGENT_ID" \
    --data-root "$ARN_DATA_ROOT" \
    --embedding-tier "$ARN_EMBEDDING_TIER" \
    "$@"
}

arn_on_message_received() {
  local role="${1:-user}"
  local message="${2:-}"
  arn_cli hook receive --role "$role" --message "$message" --importance 0.5 --time-context current
}

arn_before_reply() {
  local query="${1:-recent conversation}"
  local max_tokens="${2:-1000}"
  arn_cli hook before-reply --query "$query" --max-tokens "$max_tokens"
}

arn_preprocess_for_agent() {
  local message="${1:-}"
  local max_tokens="${2:-1000}"
  arn_cli hook preprocessed --role user --message "$message" --query "$message" --max-tokens "$max_tokens" --importance 0.6
}

arn_on_message_sent() {
  local role="${1:-assistant}"
  local message="${2:-}"
  arn_cli hook send --role "$role" --message "$message" --importance 0.4 --time-context current
}

arn_on_tool_result() {
  local tool_name="${1:-tool}"
  local result="${2:-}"
  arn_cli hook tool-result --tool-name "$tool_name" --message "$result" --importance 0.6 --time-context current
}
