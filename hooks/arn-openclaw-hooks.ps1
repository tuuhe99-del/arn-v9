# PowerShell helper functions for Windows-based local agent bridges.
# Usage:
#   . .\hooks\arn-openclaw-hooks.ps1
#   Arn-OnMessageReceived -Role user -Message "User says the site runs on port 4173"
#   $memory = Arn-BeforeReply -Query "Cloudflare tunnel setup"

if (-not $env:OPENCLAW_AGENT_ID) { $env:OPENCLAW_AGENT_ID = "default" }
if (-not $env:ARN_DATA_ROOT) { $env:ARN_DATA_ROOT = Join-Path $HOME ".arn_data" }
if (-not $env:ARN_CLI) { $env:ARN_CLI = "arn-cli" }

function Invoke-ArnCli {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & $env:ARN_CLI --strict --agent-id $env:OPENCLAW_AGENT_ID --data-root $env:ARN_DATA_ROOT @Args
}

function Arn-OnMessageReceived {
    param([string]$Role = "user", [string]$Message = "")
    Invoke-ArnCli hook receive --role $Role --message $Message --importance 0.5 --time-context current
}

function Arn-BeforeReply {
    param([string]$Query = "recent conversation", [int]$MaxTokens = 1000)
    Invoke-ArnCli hook before-reply --query $Query --max-tokens $MaxTokens
}

function Arn-PreprocessForAgent {
    param([string]$Message = "", [int]$MaxTokens = 1000)
    Invoke-ArnCli hook preprocessed --role user --message $Message --query $Message --max-tokens $MaxTokens --importance 0.6
}

function Arn-OnMessageSent {
    param([string]$Role = "assistant", [string]$Message = "")
    Invoke-ArnCli hook send --role $Role --message $Message --importance 0.4 --time-context current
}

function Arn-OnToolResult {
    param([string]$ToolName = "tool", [string]$Result = "")
    Invoke-ArnCli hook tool-result --tool-name $ToolName --message $Result --importance 0.6 --time-context current
}
