import { execFileSync } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

function eventName(event: any): string {
  if (typeof event?.event === "string") return event.event;
  if (typeof event?.name === "string") return event.name;
  if (typeof event?.type === "string" && typeof event?.action === "string") {
    return `${event.type}:${event.action}`;
  }
  return "unknown";
}

function pickContent(event: any): string {
  const ctx = event?.context || {};
  const candidates = [
    ctx.bodyForAgent,
    ctx.content,
    ctx.transcript,
    ctx.message,
    event?.message?.content,
    event?.content,
  ];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim().length > 0) return value.trim();
  }
  return "";
}

function agentId(event: any): string {
  return (
    process.env.OPENCLAW_AGENT_ID ||
    event?.context?.agentId ||
    event?.context?.agent_id ||
    event?.agentId ||
    "default"
  );
}

function homeDir(): string {
  return process.env.HOME || process.env.USERPROFILE || "/tmp";
}

function logLine(line: string) {
  const path = process.env.ARN_HOOK_LOG || `${homeDir()}/.arn_data/openclaw-hook.log`;
  try {
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(path, `${new Date().toISOString()} ${line}\n`);
  } catch (_) {
    // Never break OpenClaw because logging failed.
  }
}

function arn(args: string[], event: any, timeoutMs = 8000): string {
  const candidates = process.env.ARN_CLI
    ? [process.env.ARN_CLI]
    : (process.platform === "win32" ? ["arn.cmd", "arn-cli.cmd", "arn", "arn-cli"] : ["arn", "arn-cli"]);
  const env = {
    ...process.env,
    OPENCLAW_AGENT_ID: agentId(event),
    ARN_DATA_ROOT: process.env.ARN_DATA_ROOT || `${homeDir()}/.arn_data`,
  };
  let lastError: any = null;
  for (const bin of candidates) {
    try {
      return execFileSync(bin, args, {
        encoding: "utf8",
        timeout: timeoutMs,
        env,
        stdio: ["ignore", "pipe", "pipe"],
        shell: process.platform === "win32",
      });
    } catch (err: any) {
      lastError = err;
    }
  }
  throw lastError || new Error("ARN CLI command was not found. Install ARN or set ARN_CLI.");
}

function injectBootstrapFile(event: any, content: string) {
  const files = event?.context?.bootstrapFiles;
  if (!Array.isArray(files) || !content.trim()) return;

  // OpenClaw docs say bootstrapFiles is mutable. The exact object shape can
  // vary by version, so include common keys and keep it self-contained.
  files.push({
    name: "ARN_MEMORY.md",
    basename: "MEMORY.md",
    path: "arn://memory-context",
    content,
  });
}

const handler = async (event: any) => {
  const name = eventName(event);
  const ctx = event?.context || {};

  try {
    if (name === "gateway:startup") {
      try {
        arn(["--strict", "check"], event, 15000);
        logLine("gateway:startup arn-cli check ok");
      } catch (err: any) {
        logLine(`gateway:startup arn-cli check failed: ${err?.message || err}`);
      }
      return;
    }

    if (name === "message:received") {
      const content = pickContent(event);
      if (content) {
        arn(["hook", "receive", "--role", "user", "--message", content, "--importance", "0.55"], event);
      }
      return;
    }

    if (name === "message:preprocessed") {
      const content = pickContent(event);
      if (!content) return;

      // Store the final enriched inbound body, then retrieve relevant memory.
      arn(["hook", "receive", "--role", "user", "--message", content, "--importance", "0.6"], event);
      const memoryBlock = arn(["hook", "before-reply", "--query", content, "--max-tokens", "1000"], event);
      const trimmed = memoryBlock.trim();
      if (!trimmed) return;

      // Best current integration point for automatic pre-reply injection:
      // mutate the already preprocessed message body before it is given to the agent.
      if (typeof ctx.bodyForAgent === "string") {
        ctx.bodyForAgent = `${trimmed}\n\n---\n\n${ctx.bodyForAgent}`;
      }
      return;
    }

    if (name === "agent:bootstrap") {
      const query = ctx?.sessionEntry?.title || ctx?.agentId || "current agent context";
      const memoryBlock = arn(["hook", "before-reply", "--query", String(query), "--max-tokens", "800"], event);
      injectBootstrapFile(event, memoryBlock);
      return;
    }

    if (name === "message:sent") {
      const success = ctx.success;
      if (success === false) return;
      const content = pickContent(event);
      if (content) {
        arn(["hook", "send", "--role", "assistant", "--message", content, "--importance", "0.45"], event);
      }
      return;
    }

    if (name === "session:compact:before") {
      const marker = `OpenClaw session compaction is about to run. messageCount=${ctx.messageCount ?? "unknown"}; tokenCount=${ctx.tokenCount ?? "unknown"}.`;
      arn(["store", "--content", marker, "--importance", "0.5", "--tags", "openclaw,compaction", "--source", "system"], event);
      return;
    }
  } catch (err: any) {
    // Hooks must not throw, or they can break message delivery. Log and continue.
    logLine(`${name} failed: ${err?.message || err}`);
  }
};

export default handler;
