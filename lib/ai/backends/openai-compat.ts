import OpenAI from "openai";
import { classifyError, logLlmCall } from "../log";
import { extractJson } from "../json-util";
import type { LlmRunOptions, LlmRunResult } from "../llm";

/**
 * OpenAI-compatible backend. Reused for any provider that exposes the
 * standard `/chat/completions` endpoint: OpenAI itself, DeepSeek, MiniMax,
 * Groq, Together, OpenRouter, local LM Studio / Ollama, or a self-hosted
 * gateway such as New API / One API.
 *
 * Retry policy (hardened after the 2026-09-11 incident where the user's own
 * self-hosted gateway `newapi.fivor.dev` was intermittently unreachable in
 * the evening, surfacing as Cloudflare 502/524/530 / empty-body errors):
 *   • up to MAX_ATTEMPTS tries per base URL
 *   • exponential backoff, capped at BACKOFF_MAX_MS
 *   • honors the proxy's `Retry-After` header
 *   • only retries *transient* conditions; 401/402/403/404 are fatal
 *   • validates the 2xx body (empty / truncated JSON counts as transient)
 *
 * Failover (new in this revision): when one base URL is unreachable, the call
 * is retried against a fallback base URL (e.g. a second New API instance or a
 * different provider endpoint) before giving up. This keeps a daily run green
 * even if the primary gateway blips, without needing a totally different
 * backend. Configure via `OPENAI_BASE_URL_FALLBACK` / `DEEPSEEK_BASE_URL_-
 * FALLBACK` / `MINIMAX_BASE_URL_FALLBACK` / generic `LLM_BASE_URL_FALLBACK`
 * (comma-separated). When any fallback is present the per-URL budget is
 * tightened (fewer attempts + shorter backoff) so we fail over fast instead of
 * burning the whole job timeout on a dead endpoint.
 */
const MAX_ATTEMPTS = 5;
const ATTEMPTS_WITH_FALLBACK = 2;
const BACKOFF_BASE_MS = 2_000;
const BACKOFF_MAX_MS = 60_000;
const BACKOFF_MAX_MS_WITH_FALLBACK = 15_000;

// Cloudflare / proxy status codes that are worth retrying.
const RETRYABLE_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504, 524, 530]);

/**
 * Thrown when the HTTP call "succeeds" (2xx) but the body is unusable —
 * empty, or a truncated/incomplete JSON payload. The OpenAI-compatible
 * callers all expect JSON, so a malformed body is treated as a transient
 * upstream glitch (proxy cut the stream, etc.) and retried like a 5xx.
 */
class RetryableBackendError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RetryableBackendError";
  }
}

interface RetryDecision {
  retryable: boolean;
  waitMs: number;
}

function isRetryable(err: unknown): RetryDecision {
  if (err instanceof RetryableBackendError) {
    return { retryable: true, waitMs: BACKOFF_BASE_MS };
  }
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
  /** Optional env var holding comma-separated fallback base URLs. */
  baseUrlFallbackEnv?: string;
}

export const PRESETS: Record<OpenAICompatConfig["backend"], OpenAICompatConfig> = {
  openai: {
    backend: "openai",
    defaultBaseUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-4o-mini",
    apiKeyEnv: "OPENAI_API_KEY",
    baseUrlEnv: "OPENAI_BASE_URL",
    baseUrlFallbackEnv: "OPENAI_BASE_URL_FALLBACK",
  },
  deepseek: {
    backend: "deepseek",
    defaultBaseUrl: "https://api.deepseek.com/v1",
    // deepseek-chat alias retires 2026-07-24 — point new users at the
    // current production model instead.
    defaultModel: "deepseek-v4-flash",
    apiKeyEnv: "DEEPSEEK_API_KEY",
    baseUrlEnv: "DEEPSEEK_BASE_URL",
    baseUrlFallbackEnv: "DEEPSEEK_BASE_URL_FALLBACK",
  },
  minimax: {
    backend: "minimax",
    defaultBaseUrl: "https://api.minimax.io/v1",
    defaultModel: "MiniMax-M2.7",
    apiKeyEnv: "MINIMAX_API_KEY",
    baseUrlEnv: "MINIMAX_BASE_URL",
    baseUrlFallbackEnv: "MINIMAX_BASE_URL_FALLBACK",
  },
};

const clientCache = new Map<string, OpenAI>();

/**
 * Resolve the ordered list of base URLs to try: the configured primary first,
 * then any fallback URLs. Duplicate/empty entries are dropped.
 */
function resolveBaseUrls(cfg: OpenAICompatConfig): string[] {
  const primary =
    process.env[cfg.baseUrlEnv]?.trim() ||
    process.env.LLM_BASE_URL?.trim() ||
    cfg.defaultBaseUrl;
  const out: string[] = [primary];

  const rawFallbacks: (string | undefined)[] = [
    cfg.baseUrlFallbackEnv ? process.env[cfg.baseUrlFallbackEnv] : undefined,
    process.env.LLM_BASE_URL_FALLBACK?.trim(),
  ];
  for (const raw of rawFallbacks) {
    if (!raw) continue;
    for (const u of raw.split(",")) {
      const u2 = u.trim();
      if (u2 && !out.includes(u2)) out.push(u2);
    }
  }
  return out;
}

function getClient(baseURL: string, cfg: OpenAICompatConfig): { client: OpenAI; model: string } {
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

/**
 * Retry loop against a single base URL. Extracted so `runOpenAICompat` can
 * walk a list of base URLs and fail over on exhaustion.
 */
async function attemptBaseUrl(
  opts: LlmRunOptions,
  cfg: OpenAICompatConfig,
  baseURL: string,
  attempts: number,
  backoffMax: number,
): Promise<LlmRunResult> {
  const { client, model } = getClient(baseURL, cfg);
  const started = Date.now();
  const inputChars = opts.systemPrompt.length + opts.userPrompt.length;
  const timeoutMs = opts.timeoutMs ?? 180_000;

  let lastErr: unknown = undefined;
  for (let attempt = 1; attempt <= attempts; attempt++) {
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
      // All OpenAI-compatible callers expect JSON. A 2xx response with an
      // empty body or a truncated JSON payload (proxy cut the stream) is
      // unusable and indistinguishable from a transient glitch downstream —
      // surface it as a RetryableBackendError so the loop above re-tries
      // instead of letting JSON.parse blow up the whole run.
      const cleaned = extractJson(text);
      if (cleaned.trim() === "") {
        throw new RetryableBackendError("empty response body from model");
      }
      const truncated =
        (cleaned.startsWith("{") && !cleaned.endsWith("}")) ||
        (cleaned.startsWith("[") && !cleaned.endsWith("]"));
      if (truncated) {
        throw new RetryableBackendError("truncated/incomplete JSON response from model");
      }
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

      if (attempt === attempts) {
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
          `[openai-compat:${cfg.backend}@${baseURL}] all ${attempts} attempts failed: ${msg.slice(0, 160)}`,
        );
        throw err;
      }

      const delay = Math.min(
        waitMs || BACKOFF_BASE_MS * 2 ** (attempt - 1),
        backoffMax,
      );
      console.warn(
        `[openai-compat:${cfg.backend}@${baseURL}] attempt ${attempt}/${attempts} failed (${msg.slice(0, 80)}); retrying in ${(delay / 1000).toFixed(1)}s`,
      );
      await sleep(delay);
    }
  }
  // Unreachable: the loop either returns on success or throws on the last
  // attempt. Kept to satisfy the type checker that lastErr is assigned.
  throw lastErr;
}

export async function runOpenAICompat(
  opts: LlmRunOptions,
  cfg: OpenAICompatConfig,
): Promise<LlmRunResult> {
  const baseUrls = resolveBaseUrls(cfg);
  const hasFallback = baseUrls.length > 1;
  // With a fallback configured, fail over fast (we have somewhere to go).
  // Without one, keep the full budget like before.
  const attempts = hasFallback ? ATTEMPTS_WITH_FALLBACK : MAX_ATTEMPTS;
  const backoffMax = hasFallback ? BACKOFF_MAX_MS_WITH_FALLBACK : BACKOFF_MAX_MS;

  let lastErr: unknown = undefined;
  for (let i = 0; i < baseUrls.length; i++) {
    const baseURL = baseUrls[i];
    try {
      return await attemptBaseUrl(opts, cfg, baseURL, attempts, backoffMax);
    } catch (err) {
      lastErr = err;
      if (i === baseUrls.length - 1) {
        console.warn(
          `[openai-compat:${cfg.backend}] all ${baseUrls.length} base URL(s) exhausted; last error: ${(
            err instanceof Error ? err.message : String(err)
          ).slice(0, 160)}`,
        );
        throw err;
      }
      console.warn(
        `[openai-compat:${cfg.backend}] ${baseURL} unreachable; failing over to ${baseUrls[i + 1]}`,
      );
    }
  }
  // Unreachable: loop returns or throws. Keeps the checker happy.
  throw lastErr;
}
