/**
 * LLM backend dispatcher.
 *
 * All call sites (pipeline / enrich / trading-commentary) import `runLlm`
 * from this module instead of binding to a specific backend. The actual
 * backend is selected at runtime by the LLM_BACKEND environment variable:
 *
 *   LLM_BACKEND=claude-cli   (default; uses local Claude Code CLI, Max billing)
 *   LLM_BACKEND=anthropic    (Anthropic Messages API)
 *   LLM_BACKEND=openai       (OpenAI Chat Completions)
 *   LLM_BACKEND=deepseek     (DeepSeek, OpenAI-compatible)
 *   LLM_BACKEND=minimax      (MiniMax, OpenAI-compatible)
 *   LLM_BACKEND=zhipu        (Zhipu AI / 智谱, Anthropic-compatible)
 *
 * Per-backend config (API keys, models, base URLs) lives in .env.local.
 * See .env.example for the full list.
 */

import { CLAUDE_MODEL, runClaudeCli } from "./backends/claude-cli";
import {
  PRESETS as ANTHROPIC_PRESETS,
  anthropicCompatModel,
  runAnthropicCompat,
} from "./backends/anthropic-compat";
import {
  PRESETS as OPENAI_PRESETS,
  openaiCompatModel,
  runOpenAICompat,
} from "./backends/openai-compat";

export interface LlmRunOptions {
  systemPrompt: string;
  userPrompt: string;
  timeoutMs?: number;
}

export interface LlmRunResult {
  text: string;
  durationMs: number;
}

export type LlmBackendId =
  | "claude-cli"
  | "anthropic"
  | "openai"
  | "deepseek"
  | "minimax"
  | "zhipu";

const VALID_BACKENDS: ReadonlySet<LlmBackendId> = new Set([
  "claude-cli",
  "anthropic",
  "openai",
  "deepseek",
  "minimax",
  "zhipu",
]);

export function getBackend(): LlmBackendId {
  const raw = (process.env.LLM_BACKEND?.trim() || "claude-cli").toLowerCase();
  if (!VALID_BACKENDS.has(raw as LlmBackendId)) {
    throw new Error(
      `Unknown LLM_BACKEND='${raw}'. Valid values: ${[...VALID_BACKENDS].join(", ")}`,
    );
  }
  return raw as LlmBackendId;
}

/**
 * Returns the active model name for the configured backend, useful for
 * stamping a MODEL_TAG into report metadata.
 */
function getActiveModel(backend: LlmBackendId): string {
  switch (backend) {
    case "claude-cli":
      return CLAUDE_MODEL;
    case "anthropic":
    case "zhipu":
      return anthropicCompatModel(ANTHROPIC_PRESETS[backend]);
    case "openai":
    case "deepseek":
    case "minimax":
      return openaiCompatModel(OPENAI_PRESETS[backend]);
  }
}

export function getModelTag(): string {
  const backend = getBackend();
  return `${backend}-${getActiveModel(backend)}`;
}

/**
 * Does the given backend have usable credentials in the environment? Used to
 * skip fallback backends that would just error out on a missing key, so a
 * failover chain only ever tries backends the user actually configured.
 */
function backendHasCredentials(backend: LlmBackendId): boolean {
  if (backend === "claude-cli") return true;
  const required: Record<Exclude<LlmBackendId, "claude-cli">, string> = {
    anthropic: "ANTHROPIC_API_KEY",
    openai: "OPENAI_API_KEY",
    deepseek: "DEEPSEEK_API_KEY",
    minimax: "MINIMAX_API_KEY",
    zhipu: "ZHIPU_API_KEY",
  };
  return !!(process.env[required[backend]] || process.env.LLM_API_KEY);
}

/**
 * Build the ordered backend chain: the configured primary first, then any
 * backends listed in `LLM_BACKEND_FALLBACK` (comma-separated) that have
 * credentials present. Backends without keys are skipped with a warning
 * rather than attempted. The result is always at least [primary].
 */
function buildBackendChain(): LlmBackendId[] {
  const primary = getBackend();
  const chain: LlmBackendId[] = [primary];
  const fbRaw = (process.env.LLM_BACKEND_FALLBACK?.trim() || "").toLowerCase();
  if (!fbRaw) return chain;
  for (const b of fbRaw.split(",")) {
    const b2 = b.trim() as LlmBackendId;
    if (!VALID_BACKENDS.has(b2) || chain.includes(b2)) continue;
    if (backendHasCredentials(b2)) {
      chain.push(b2);
    } else {
      console.warn(
        `[llm] skipping fallback backend '${b2}' — no credentials present; set its key or remove it from LLM_BACKEND_FALLBACK`,
      );
    }
  }
  return chain;
}

export async function runLlm(opts: LlmRunOptions): Promise<LlmRunResult> {
  const chain = buildBackendChain();
  let lastErr: unknown;
  for (let i = 0; i < chain.length; i++) {
    const backend = chain[i];
    try {
      switch (backend) {
        case "claude-cli":
          return await runClaudeCli(opts);
        case "anthropic":
        case "zhipu":
          return await runAnthropicCompat(opts, ANTHROPIC_PRESETS[backend]);
        case "openai":
        case "deepseek":
        case "minimax":
          return await runOpenAICompat(opts, OPENAI_PRESETS[backend]);
      }
    } catch (err) {
      lastErr = err;
      if (i === chain.length - 1) {
        console.warn(
          `[llm] all ${chain.length} backend(s) in chain exhausted; last error: ${(
            err instanceof Error ? err.message : String(err)
          ).slice(0, 160)}`,
        );
        throw err;
      }
      console.warn(
        `[llm] backend '${backend}' failed; failing over to '${chain[i + 1]}'`,
      );
    }
  }
  // Unreachable: the loop returns on success or throws on the last backend.
  throw lastErr;
}
/**
 * Cheap startup sanity-check so a misconfigured backend errors in <1s
 * instead of after 30s of source-fetching + half a dozen confusing
 * "ANTHROPIC_API_KEY required" lines deep into the pipeline.
 *
 * The default LLM_BACKEND in the GH Actions workflow is `deepseek`,
 * so the most common forker mistake is: add ANTHROPIC_API_KEY as a
 * secret, forget to add the matching `LLM_BACKEND=anthropic` variable,
 * then watch the run blow up looking for a key they never intended
 * to use. We detect that exact case and tell them how to fix it.
 */
export function validateBackendCredentials(): void {
  const backend = getBackend();
  if (backend === "claude-cli") return;

  const required: Record<Exclude<LlmBackendId, "claude-cli">, string> = {
    anthropic: "ANTHROPIC_API_KEY",
    openai: "OPENAI_API_KEY",
    deepseek: "DEEPSEEK_API_KEY",
    minimax: "MINIMAX_API_KEY",
    zhipu: "ZHIPU_API_KEY",
  };
  const requiredVar = required[backend];

  if (process.env[requiredVar] || process.env.LLM_API_KEY) return;

  const otherKeysSet = Object.entries(required)
    .filter(([b, v]) => b !== backend && !!process.env[v])
    .map(([b, v]) => ({ backend: b, var: v }));

  const lines: string[] = [
    `LLM_BACKEND=${backend} but ${requiredVar} (and generic LLM_API_KEY) are both unset.`,
  ];
  if (otherKeysSet.length > 0) {
    lines.push(
      "",
      "Other API keys ARE present in the environment — likely you meant to use one of those:",
    );
    for (const k of otherKeysSet) {
      lines.push(`  • ${k.var} is set → switch to LLM_BACKEND=${k.backend}`);
    }
    lines.push(
      "",
      "Fix one of:",
      `  (a) set LLM_BACKEND to match the key you actually have, or`,
      `  (b) add ${requiredVar} for the backend you currently selected.`,
    );
  } else {
    lines.push(
      "",
      `Fix: set ${requiredVar} (or the generic LLM_API_KEY).`,
    );
  }
  lines.push(
    "",
    "Where to set it:",
    "  • Local:          .env.local at the repo root",
    "  • GitHub Actions: Settings → Secrets and variables → Actions",
    "                    (Secrets tab for the API key, Variables tab for LLM_BACKEND)",
  );
  throw new Error(lines.join("\n"));
}
