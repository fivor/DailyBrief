import OpenAI from "openai";
import type { ChatCompletionCreateParamsNonStreaming } from "openai/resources/chat/completions";
import { classifyError, logLlmCall } from "../log";
import { extractJson } from "../json-util";
import type { LlmRunOptions, LlmRunResult } from "../llm";

/**
 * OpenAI-compatible backend. Reused for any provider that exposes the
 * standard `/chat/completions` endpoint: OpenAI itself, DeepSeek, MiniMax,
 * Groq, Together, OpenRouter, local LM Studio / Ollama, or a self-hosted
 * gateway such as New API / One API.
 *
 * Retry policy (hardened across the 2026-09-11 → 09-12 incidents):
 *   • up to MAX_ATTEMPTS tries per endpoint
 *   • exponential backoff, capped at BACKOFF_MAX_MS
 *   • honors the proxy's `Retry-After` header *in full* (up to
 *     RETRY_AFTER_MAX_MS) — a per-minute rate limit needs a long wait, and
 *     failing over to another endpoint on the same upstream quota does not
 *     escape it
 *   • only retries *transient* conditions; 401/402/403/404 are fatal for that
 *     endpoint (the failover layer below may still recover)
 *   • validates the 2xx body (empty / truncated JSON counts as transient)
 *
 * Reasoning-budget guard (2026-09-12): some models emit a long chain-of-
 * thought that shares the output budget with `content`. On big tasks the
 * reasoning can eat the whole `max_tokens` and leave `content` empty
 * ("empty response body from model"), while the reasoning tokens also blow
 * per-minute quota. Callers can tame this via env (applied to every request):
 *   LLM_REASONING_EFFORT=none|minimal|low|medium|high  → `reasoning_effort`
 *   LLM_EXTRA_BODY='{"chat_template_kwargs":{...}}'    → merged verbatim
 *
 * Failover: when one endpoint is unreachable, the call is retried against a
 * fallback endpoint (e.g. the provider's direct API instead of a flaky
 * self-hosted gateway). Configure via LLM_BASE_URL_FALLBACK /
 * OPENAI_BASE_URL_FALLBACK (and LLM_API_KEY_FALLBACK / OPENAI_API_KEY_FALLBACK
 * for a fallback-specific key). Each is comma-separated for multiple values.
 */
const MAX_ATTEMPTS = 5;
const ATTEMPTS_WITH_FALLBACK = 3;
const BACKOFF_BASE_MS = 2_000;
const BACKOFF_MAX_MS = 60_000;
const BACKOFF_MAX_MS_WITH_FALLBACK = 30_000;
const RETRY_AFTER_MAX_MS = 120_000;
/** Output token budget. With reasoning disabled this is content-only. */
const MAX_OUTPUT_TOKENS = 8192;

// Cloudflare / proxy status codes that are worth retrying.
const RETRYABLE_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504, 524, 530]);

/**
 * Thrown when the HTTP call "succeeds" (2xx) but the body is unusable —
 * empty, or a truncated/incomplete JSON payload. The OpenAI-compatible
 * callers all expect JSON, so a malformed body is treated as a transient
 * upstream glitch (proxy cut the stream, reasoning ate the budget, ...) and
 * retried like a 5xx.
 */
class RetryableBackendError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RetryableBackendError";
  }
}

interface RetryDecision {
  retryable: boolean;
  /** Wait the server explicitly asked for (Retry-After), if any. */
  serverWaitMs?: number;
}

function isRetryable(err: unknown): RetryDecision {
  if (err instanceof RetryableBackendError) {
    return { retryable: true };
  }
  const msg = err instanceof Error ? err.message : String(err ?? "");

  // Truncated / malformed responses from a flaky proxy — transient.
  if (/Unexpected end of JSON input/i.test(msg)) {
    return { retryable: true };
  }

  // SDK error objects expose a numeric `status`; fall back to scraping the
  // message ("524 status code") which is how Cloudflare-proxied errors read.
  const m = msg.match(/\b(\d{3})\b status code/);
  const statusFromMsg = m ? Number(m[1]) : undefined;
  const status = (err as { status?: unknown } | null)?.status ?? statusFromMsg;

  if (typeof status === "number") {
    // 4xx (except the retryable ones above) are client errors — not worth
    // retrying against the same endpoint; surface them so the failover layer
    // can try the next one or a bad key / missing payment gets fixed.
    if (status >= 400 && status < 500 && !RETRYABLE_STATUS.has(status)) {
      return { retryable: false };
    }
    if (RETRYABLE_STATUS.has(status)) {
      const headers = (err as { headers?: { get?: (k: string) => string | null } } | null)
        ?.headers;
      const ra = headers?.get?.("retry-after");
      if (ra) {
        const secs = Number(ra);
        if (!Number.isNaN(secs)) {
          return { retryable: true, serverWaitMs: secs * 1000 };
        }
      }
      return { retryable: true };
    }
  }

  // Network-level failures (connection reset, timeout, DNS, socket hang up,
  // or the OpenAI SDK's generic "Connection error." on a failed fetch).
  if (
    /ECONNRESET|ETIMEDOUT|ECONNREFUSED|Connection error|socket hang up|fetch failed|ENOTFOUND|network|timed out|timeout/i.test(
      msg,
    )
  ) {
    return { retryable: true };
  }

  return { retryable: false };
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
  /** Optional env var holding the API key used for fallback endpoints. */
  apiKeyFallbackEnv?: string;
}

export const PRESETS: Record<OpenAICompatConfig["backend"], OpenAICompatConfig> = {
  openai: {
    backend: "openai",
    defaultBaseUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-4o-mini",
    apiKeyEnv: "OPENAI_API_KEY",
    baseUrlEnv: "OPENAI_BASE_URL",
    baseUrlFallbackEnv: "OPENAI_BASE_URL_FALLBACK",
    apiKeyFallbackEnv: "OPENAI_API_KEY_FALLBACK",
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
    apiKeyFallbackEnv: "DEEPSEEK_API_KEY_FALLBACK",
  },
  minimax: {
    backend: "minimax",
    defaultBaseUrl: "https://api.minimax.io/v1",
    defaultModel: "MiniMax-M2.7",
    apiKeyEnv: "MINIMAX_API_KEY",
    baseUrlEnv: "MINIMAX_BASE_URL",
    baseUrlFallbackEnv: "MINIMAX_BASE_URL_FALLBACK",
    apiKeyFallbackEnv: "MINIMAX_API_KEY_FALLBACK",
  },
};

const clientCache = new Map<string, OpenAI>();

interface Endpoint {
  baseURL: string;
  apiKey: string;
  label: "primary" | "fallback";
}

/**
 * Resolve the ordered list of endpoints to try: the configured primary first,
 * then any fallback URLs. Each fallback URL uses the fallback key when set,
 * otherwise the primary key. Duplicate/empty entries are dropped.
 */
function resolveEndpoints(cfg: OpenAICompatConfig): Endpoint[] {
  const primaryKey = process.env[cfg.apiKeyEnv] || process.env.LLM_API_KEY;
  if (!primaryKey) {
    throw new Error(
      `${cfg.apiKeyEnv} (or generic LLM_API_KEY) is required for LLM_BACKEND=${cfg.backend}. Set it in .env.local.`,
    );
  }
  const primaryUrl =
    process.env[cfg.baseUrlEnv]?.trim() ||
    process.env.LLM_BASE_URL?.trim() ||
    cfg.defaultBaseUrl;
  const endpoints: Endpoint[] = [
    { baseURL: primaryUrl, apiKey: primaryKey, label: "primary" },
  ];

  const fallbackKey =
    (cfg.apiKeyFallbackEnv ? process.env[cfg.apiKeyFallbackEnv]?.trim() : undefined) ||
    process.env.LLM_API_KEY_FALLBACK?.trim() ||
    primaryKey;

  const rawFallbacks: (string | undefined)[] = [
    cfg.baseUrlFallbackEnv ? process.env[cfg.baseUrlFallbackEnv] : undefined,
    process.env.LLM_BASE_URL_FALLBACK?.trim(),
  ];
  for (const raw of rawFallbacks) {
    if (!raw) continue;
    for (const u of raw.split(",")) {
      const u2 = u.trim();
      if (u2 && !endpoints.some((e) => e.baseURL === u2)) {
        endpoints.push({ baseURL: u2, apiKey: fallbackKey, label: "fallback" });
      }
    }
  }
  return endpoints;
}

/**
 * Extra request-body fields applied to every call. Lets the deployment tame a
 * runaway chain-of-thought (which shares the output budget and can starve
 * `content`) without a code change.
 */
function extraBodyParams(): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const effort = process.env.LLM_REASONING_EFFORT?.trim();
  if (effort) out.reasoning_effort = effort;
  const raw = process.env.LLM_EXTRA_BODY?.trim();
  if (raw) {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        Object.assign(out, parsed);
      } else {
        console.warn("[openai-compat] LLM_EXTRA_BODY must be a JSON object; ignoring");
      }
    } catch {
      console.warn("[openai-compat] LLM_EXTRA_BODY is not valid JSON; ignoring");
    }
  }
  return out;
}

function getClient(
  baseURL: string,
  apiKey: string,
  cfg: OpenAICompatConfig,
): { client: OpenAI; model: string } {
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
 * Retry loop against a single endpoint. Extracted so `runOpenAICompat` can
 * walk a list of endpoints and fail over on exhaustion.
 */
async function attemptEndpoint(
  opts: LlmRunOptions,
  cfg: OpenAICompatConfig,
  endpoint: Endpoint,
  attempts: number,
  backoffMax: number,
): Promise<LlmRunResult> {
  const { client, model } = getClient(endpoint.baseURL, endpoint.apiKey, cfg);
  const started = Date.now();
  const inputChars = opts.systemPrompt.length + opts.userPrompt.length;
  const timeoutMs = opts.timeoutMs ?? 180_000;
  const tag = `${cfg.backend}@${endpoint.baseURL}`;

  const extraBody = extraBodyParams();
  const body: ChatCompletionCreateParamsNonStreaming = {
    model,
    messages: [
      { role: "system", content: opts.systemPrompt },
      { role: "user", content: opts.userPrompt },
    ],
    // Explicit max_tokens — most providers default low (DeepSeek 4096, some
    // MiniMax variants 2048). 8192 covers all observed daily batches with
    // headroom once reasoning is tamed. Match the Anthropic SDK's value.
    max_tokens: MAX_OUTPUT_TOKENS,
    // Don't force JSON mode — not all OpenAI-compat providers support
    // response_format=json_object, and our prompts + jsonrepair handle slop.
  };
  // Merge deployment-level extras (reasoning_effort, provider quirks) after
  // the typed core so unknown keys are still serialized onto the wire.
  Object.assign(body, extraBody);

  let lastErr: unknown = undefined;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const resp = await client.chat.completions.create(body, { timeout: timeoutMs });
      const finishReason = resp.choices[0]?.finish_reason ?? "unknown";
      const text = (resp.choices[0]?.message?.content ?? "").trim();
      // All OpenAI-compatible callers expect JSON. A 2xx response with an
      // empty body or a truncated JSON payload (proxy cut the stream, or a
      // reasoning model that spent the whole budget on chain-of-thought) is
      // unusable and indistinguishable from a transient glitch downstream —
      // surface it as a RetryableBackendError so the loop above re-tries
      // instead of letting JSON.parse blow up the whole run.
      const cleaned = extractJson(text);
      if (cleaned.trim() === "") {
        throw new RetryableBackendError(
          `empty response body from model (finish_reason=${finishReason})`,
        );
      }
      const truncated =
        (cleaned.startsWith("{") && !cleaned.endsWith("}")) ||
        (cleaned.startsWith("[") && !cleaned.endsWith("]"));
      if (truncated) {
        throw new RetryableBackendError(
          `truncated/incomplete JSON response from model (finish_reason=${finishReason})`,
        );
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
      const { retryable, serverWaitMs } = isRetryable(err);
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
        console.warn(`[openai-compat:${tag}] non-retryable error (${msg.slice(0, 160)})`);
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
          `[openai-compat:${tag}] all ${attempts} attempts failed: ${msg.slice(0, 160)}`,
        );
        throw err;
      }

      // Honor the server's Retry-After in full (a TPM/RPM limit needs the
      // long wait; falling over to a shared-quota endpoint won't escape it).
      // Otherwise use exponential backoff bounded by the endpoint budget.
      const expo = Math.min(BACKOFF_BASE_MS * 2 ** (attempt - 1), backoffMax);
      const delay =
        serverWaitMs != null ? Math.min(serverWaitMs, RETRY_AFTER_MAX_MS) : expo;
      console.warn(
        `[openai-compat:${tag}] attempt ${attempt}/${attempts} failed (${msg.slice(0, 80)}); retrying in ${(delay / 1000).toFixed(1)}s`,
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
  const endpoints = resolveEndpoints(cfg);
  const hasFallback = endpoints.length > 1;
  // With a fallback configured, fail over faster (we have somewhere to go).
  // Without one, keep the full budget like before.
  const attempts = hasFallback ? ATTEMPTS_WITH_FALLBACK : MAX_ATTEMPTS;
  const backoffMax = hasFallback ? BACKOFF_MAX_MS_WITH_FALLBACK : BACKOFF_MAX_MS;

  let lastErr: unknown = undefined;
  for (let i = 0; i < endpoints.length; i++) {
    const endpoint = endpoints[i];
    try {
      return await attemptEndpoint(opts, cfg, endpoint, attempts, backoffMax);
    } catch (err) {
      lastErr = err;
      if (i === endpoints.length - 1) {
        console.warn(
          `[openai-compat:${cfg.backend}] all ${endpoints.length} endpoint(s) exhausted; last error: ${(
            err instanceof Error ? err.message : String(err)
          ).slice(0, 160)}`,
        );
        throw err;
      }
      console.warn(
        `[openai-compat:${cfg.backend}] ${endpoint.label} endpoint ${endpoint.baseURL} unreachable; failing over to ${endpoints[i + 1].baseURL}`,
      );
    }
  }
  // Unreachable: loop returns or throws. Keeps the checker happy.
  throw lastErr;
}
