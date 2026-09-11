import OpenAI from "openai";
import { classifyError, logLlmCall } from "../log";
import type { LlmRunOptions, LlmRunResult } from "../llm";

/**
 * OpenAI-compatible backend. Reused for any provider that exposes the
 * standard `/chat/completions` endpoint: OpenAI itself, DeepSeek, MiniMax,
 * Groq, Together, OpenRouter, local LM Studio / Ollama, etc.
 *
 * Retry policy (added after the 2026-09-11 incident where transient
 * provider / Cloudflare errors — 429 rate-limit, 524 origin timeout,
 * 530 origin unreachable, truncated-JSON — failed whole daily runs):
 *   • up to MAX_ATTEMPTS tries
 *   • exponential backoff, capped at BACKOFF_MAX_MS
 *   • honors the proxy's `Retry-After` header (e.g. sensenova/Cloudflare
 *     returns `retry-after: 120` on 429)
 *   • only retries *transient* conditions; 401/402/403/404 are fatal and
 *     surfaced immediately so a config problem isn't masked by retries
 */
const MAX_ATTEMPTS = 5;
const BACKOFF_BASE_MS = 2_000;
const BACKOFF_MAX_MS = 60_000;

// Cloudflare / proxy status codes that are worth retrying.
const RETRYABLE_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504, 524, 530]);

interface RetryDecision {
  retryable: boolean;
  waitMs: number;
}

function isRetryable(err: unknown): RetryDecision {
  const msg = err instanceof Error ? err.message : String(err ?? "");

  // Truncated / malformed responses from a flaky proxy — transient.
  if (/Unexpected end of JSON input/i.test(msg)) {
    return { retryable: true, waitMs: BACKOFF_BASE_MS };
  }

  // SDK error objects expose a numeric `status`; fall back to scraping the
  // message ("524 status code") which is how Cloudflare-proxied errors read.
  const m = msg.match(/\b(\d{3})\b status code/);
  const statusFromMsg = m ? Number(m[1]) : undefined;
  const status = (err as { status?: unknown } | null)?.status ?? statusFromMsg;

  if (typeof status === "number") {
    // 4xx (except the retryable ones above) are client errors — not worth
    // retrying; surface them so a bad key / missing payment gets fixed.
    if (status >= 400 && status < 500 && !RETRYABLE_STATUS.has(status)) {
      return { retryable: false, waitMs: 0 };
    }
    if (RETRYABLE_STATUS.has(status)) {
      const headers = (err as { headers?: { get?: (k: string) => string | null } } | null)
        ?.headers;
      const ra = headers?.get?.("retry-after");
      let waitMs = BACKOFF_BASE_MS;
      if (ra) {
        const secs = Number(ra);
        if (!Number.isNaN(secs)) waitMs = Math.min(secs * 1000, BACKOFF_MAX_MS);
      }
      return { retryable: true, waitMs };
    }
  }

  // Network-level failures (connection reset, timeout, DNS, socket hang up).
  if (
    /ECONNRESET|ETIMEDOUT|ECONNREFUSED|socket hang up|fetch failed|ENOTFOUND|network|timed out|timeout/i.test(
      msg,
    )
  ) {
    return { retryable: true, waitMs: BACKOFF_BASE_MS };
  }

  return { retryable: false, waitMs: 0 };
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export interface OpenAICompatConfig {
  /** Stable backend id, used in logs and error messages */
  backend: "openai" | "deepseek" | "minimax";
  defaultBaseUrl: string;
  defaultModel: string;
  apiKeyEnv: string;
  baseUrlEnv: string;
}

export const PRESETS: Record<OpenAICompatConfig["backend"], OpenAICompatConfig> = {
  openai: {
    backend: "openai",
    defaultBaseUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-4o-mini",
    apiKeyEnv: "OPENAI_API_KEY",
    baseUrlEnv: "OPENAI_BASE_URL",
  },
  deepseek: {
    backend: "deepseek",
    defaultBaseUrl: "https://api.deepseek.com/v1",
    // deepseek-chat alias retires 2026-07-24 — point new users at the
    // current production model instead.
    defaultModel: "deepseek-v4-flash",
    apiKeyEnv: "DEEPSEEK_API_KEY",
    baseUrlEnv: "DEEPSEEK_BASE_URL",
  },
  minimax: {
    backend: "minimax",
    defaultBaseUrl: "https://api.minimax.io/v1",
    defaultModel: "MiniMax-M2.7",
    apiKeyEnv: "MINIMAX_API_KEY",
    baseUrlEnv: "MINIMAX_BASE_URL",
  },
};

const clientCache = new Map<string, OpenAI>();

function getClient(cfg: OpenAICompatConfig): { client: OpenAI; model: string } {
  // Provider-specific env wins; LLM_API_KEY / LLM_BASE_URL are generic
  // aliases so users pointing at a non-preset OpenAI-compatible service
  // (Moonshot, SiliconFlow, OpenRouter, self-hosted vLLM, ...) don't have
  // to misuse the OPENAI_* variable names just to reach a custom endpoint.
  const apiKey = process.env[cfg.apiKeyEnv] || process.env.LLM_API_KEY;
  if (!apiKey) {
    throw new Error(
      `${cfg.apiKeyEnv} (or generic LLM_API_KEY) is required for LLM_BACKEND=${cfg.backend}. Set it in .env.local.`,
    );
  }
  const baseURL =
    process.env[cfg.baseUrlEnv]?.trim() ||
    process.env.LLM_BASE_URL?.trim() ||
    cfg.defaultBaseUrl;
  const model = process.env.LLM_MODEL?.trim() || cfg.defaultModel;

  const cacheKey = `${baseURL}::${apiKey.slice(-6)}`;
  let client = clientCache.get(cacheKey);
  if (!client) {
    // maxRetries: 0 — we run our own retry loop below so backoff + Retry-After
    // handling is centralized and consistent across all call sites.
    client = new OpenAI({ apiKey, baseURL, maxRetries: 0 });
    clientCache.set(cacheKey, client);
  }
  return { client, model };
}

export function openaiCompatModel(cfg: OpenAICompatConfig): string {
  return process.env.LLM_MODEL?.trim() || cfg.defaultModel;
}

export async function runOpenAICompat(
  opts: LlmRunOptions,
  cfg: OpenAICompatConfig,
): Promise<LlmRunResult> {
  const { client, model } = getClient(cfg);
  const started = Date.now();
  const inputChars = opts.systemPrompt.length + opts.userPrompt.length;
  const timeoutMs = opts.timeoutMs ?? 180_000;

  let lastErr: unknown = undefined;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const resp = await client.chat.completions.create(
        {
          model,
          messages: [
            { role: "system", content: opts.systemPrompt },
            { role: "user", content: opts.userPrompt },
          ],
          // Explicit max_tokens — most providers default low (DeepSeek 4096,
          // some MiniMax variants 2048). A 16-item batch enrichment routinely
          // exceeds 4K output tokens once you count Chinese chars + JSON
          // structure, and silent truncation made it through with just 1/16
          // entries parseable. 8192 covers all observed daily batches with
          // generous headroom. Match the explicit value Anthropic SDK uses.
          max_tokens: 8192,
          // Don't force JSON mode — not all OpenAI-compat providers support
          // response_format=json_object, and our prompts + jsonrepair already
          // handle the slop.
        },
        { timeout: timeoutMs },
      );
      const text = (resp.choices[0]?.message?.content ?? "").trim();
      const durationMs = Date.now() - started;
      logLlmCall({
        ts: new Date(started).toISOString(),
        backend: cfg.backend,
        model,
        durationMs,
        success: true,
        inputChars,
        outputChars: text.length,
        errorCategory: null,
        errorSnippet: null,
      });
      return { text, durationMs };
    } catch (err) {
      lastErr = err;
      const { retryable, waitMs } = isRetryable(err);
      const msg = err instanceof Error ? err.message : String(err);

      if (!retryable) {
        const durationMs = Date.now() - started;
        logLlmCall({
          ts: new Date(started).toISOString(),
          backend: cfg.backend,
          model,
          durationMs,
          success: false,
          inputChars,
          outputChars: 0,
          errorCategory: classifyError(msg),
          errorSnippet: msg.slice(0, 200),
        });
        console.warn(
          `[openai-compat:${cfg.backend}] non-retryable error (${msg.slice(0, 160)})`,
        );
        throw err;
      }

      if (attempt === MAX_ATTEMPTS) {
        const durationMs = Date.now() - started;
        logLlmCall({
          ts: new Date(started).toISOString(),
          backend: cfg.backend,
          model,
          durationMs,
          success: false,
          inputChars,
          outputChars: 0,
          errorCategory: classifyError(msg),
          errorSnippet: msg.slice(0, 200),
        });
        console.warn(
          `[openai-compat:${cfg.backend}] all ${MAX_ATTEMPTS} attempts failed: ${msg.slice(0, 160)}`,
        );
        throw err;
      }

      const delay = Math.min(
        waitMs || BACKOFF_BASE_MS * 2 ** (attempt - 1),
        BACKOFF_MAX_MS,
      );
      console.warn(
        `[openai-compat:${cfg.backend}] attempt ${attempt}/${MAX_ATTEMPTS} failed (${msg.slice(0, 80)}); retrying in ${(delay / 1000).toFixed(1)}s`,
      );
      await sleep(delay);
    }
  }
  // Unreachable: the loop either returns on success or throws on the last
  // attempt. Kept to satisfy the type checker that lastErr is assigned.
  throw lastErr;
}
