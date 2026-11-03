import { jsonrepair } from "jsonrepair";
import { runLlm } from "./llm";
import { extractJson } from "./json-util";
import { SYSTEM_PROMPT_DIGEST_EN, SYSTEM_PROMPT_DIGEST_ZH } from "./prompts";
import { REPORT_LOCALE } from "../sources/registry";
import type { Category, RawArticle } from "../sources/types";

const SYSTEM_PROMPT_DIGEST =
  REPORT_LOCALE === "en" ? SYSTEM_PROMPT_DIGEST_EN : SYSTEM_PROMPT_DIGEST_ZH;

export interface BriefItem {
  title: string;
  url: string;
  source: string;
  summary: string;
  importance: number;
}

export interface DailyReport {
  hero_headline: string;
  daily_overview: string;
  tech_briefs: BriefItem[];
  finance_briefs: BriefItem[];
  politics_briefs: BriefItem[];
  editor_note: string;
  keywords: string[];
  /** Optional trading-signals section, present when scripts/daily.ts ran successfully. */
  trading?: TradingSection;
}

import type { TickerAnalysis } from "../trading/signals";
import type { CryptoGlobalStats } from "../trading/coingecko";
import type { FearGreedSnapshot } from "../trading/fear-greed";
import type { TradingCommentary } from "./trading-commentary";

export interface TradingSection extends TradingCommentary {
  generated_at: string;
  tickers: TickerAnalysis[];
  crypto_fear_greed?: FearGreedSnapshot;
  crypto_global?: CryptoGlobalStats;
}

export interface ArticleInput extends RawArticle {
  source: string;
}

const PER_CATEGORY_LIMIT: Record<Category, number> = {
  tech: 25,
  finance: 20,
  politics: 15,
};

const MAX_AGE_DAYS = 14;

/**
 * Pick `limit` items from `items` so every source gets a fair shot.
 *
 * Why this exists: the previous `slice(0, limit)` honored insertion order,
 * which is the source-iteration order in daily.ts. That gave whichever
 * source came first 100% of the quota — e.g. all 25 tech slots filled by
 * Hacker News before GitHub Trending / Solidot / V2EX / 阮一峰 got a turn.
 *
 * Strategy: drop items older than MAX_AGE_DAYS, group by sourceId,
 * sort each bucket newest-first, then round-robin one item per source
 * until we hit the limit. Sources with fewer items naturally drop out
 * and others absorb the slack.
 */
function selectRoundRobin(
  items: ArticleInput[],
  limit: number,
): ArticleInput[] {
  const cutoff = Date.now() - MAX_AGE_DAYS * 86_400_000;
  const fresh = items.filter(
    (it) => !it.publishedAt || it.publishedAt.getTime() >= cutoff,
  );

  const bySource = new Map<string, ArticleInput[]>();
  for (const it of fresh) {
    const arr = bySource.get(it.sourceId) ?? [];
    arr.push(it);
    bySource.set(it.sourceId, arr);
  }
  for (const arr of bySource.values()) {
    arr.sort(
      (a, b) =>
        (b.publishedAt?.getTime() ?? 0) - (a.publishedAt?.getTime() ?? 0),
    );
  }

  const buckets = Array.from(bySource.values());
  const out: ArticleInput[] = [];
  let madeProgress = true;
  while (out.length < limit && madeProgress) {
    madeProgress = false;
    for (const b of buckets) {
      if (b.length === 0) continue;
      out.push(b.shift()!);
      madeProgress = true;
      if (out.length >= limit) break;
    }
  }
  return out;
}

export async function callOnce(userPayloadJson: string): Promise<DailyReport> {
  // Claude Code CLI's built-in system prompt biases the model toward
  // conversational markdown output. Anchor the format expectation in the
  // user message (instruction recency wins) *and* explicitly demand every
  // schema field be populated — without this Sonnet has been observed to
  // emit a JSON shell with empty arrays to "satisfy" a JSON-only ask.
  const userPrompt =
    REPORT_LOCALE === "en"
      ? [
          "**Output language: ENGLISH ONLY.** Every string value in the JSON — hero_headline, daily_overview, every brief's title/summary, editor_note, keywords — must be written entirely in English. No Chinese characters anywhere.",
          "",
          "Your task: generate today's daily brief from the candidate news below. **The response MUST be a single valid JSON object** — starts with `{`, ends with `}`, no markdown, no code fences, no explanations.",
          "",
          "The JSON must contain every field non-empty (briefs arrays per the system-prompt counts):",
          "  - hero_headline: 10-25 word headline of the day",
          "  - daily_overview: **150-250 word** paragraph covering tech / finance / politics signals so a reader sees the whole picture at a glance",
          "  - tech_briefs: **3-5** tech BriefItems",
          "  - finance_briefs: **3-5** finance BriefItems",
          "  - politics_briefs: **2-3** politics BriefItems",
          "  - editor_note: 30-60 word editor's note",
          "  - keywords: 5-8 keywords",
          "",
          "BriefItem fields: title, url (copied verbatim from candidate), source, summary, importance (1-10).",
          "**Quote rule (important!)**: For any quotation INSIDE a JSON string, use single quotes ' or curly quotes '\" — **never** raw double quotes \", which break JSON parsing.",
          "No trailing commas.",
          "",
          `Candidate news (JSON array, ${userPayloadJson.length} chars):`,
          userPayloadJson,
        ].join("\n")
      : [
          "你的任务：根据下方候选新闻，生成一份当日简报，**响应必须是一个合法 JSON 对象**——以 `{` 开头，以 `}` 结尾，不要 markdown / 不要代码围栏 / 不要任何解释。",
          "",
          "JSON 必须包含全部字段且不能为空（briefs 数组按 system prompt 规定的条数填充）：",
          "  - hero_headline: 10-25 字的当日一句话头条",
          "  - daily_overview: **150-220 字** 的当日总览段落，一段话覆盖技术 / 财经 / 时政 的核心信号，让读者一眼抓住全貌",
          "  - tech_briefs: **3-5 条** 科技 BriefItem",
          "  - finance_briefs: **3-5 条** 财经 BriefItem",
          "  - politics_briefs: **2-3 条** 时政 BriefItem",
          "  - editor_note: 30-60 字的编辑短评",
          "  - keywords: 5-8 个关键词",
          "",
          "BriefItem 字段：title、url（必须从候选条目原样选取）、source、summary、importance(1-10)。",
          "**引号规则（重要！）**：JSON 字符串内的中文引用请使用**中文全角引号**「」或者 “”，**绝对不要**用英文双引号 \" —— 那会导致 JSON 解析失败。例：写 商务部回应「内卷」 而不是 商务部回应\"内卷\"。",
          "不要使用单引号、不要末尾多余逗号。",
          "",
          "候选新闻（JSON 数组，共 " + userPayloadJson.length + " 字符）：",
          userPayloadJson,
        ].join("\n");
  if (process.env.DAILY_DEBUG_PROMPT) {
    console.log("[pipeline] digest userPrompt:\n" + userPrompt);
  }
  const { text } = await runLlm({
    systemPrompt: SYSTEM_PROMPT_DIGEST,
    userPrompt,
  });
  const cleaned = extractJson(text);
  let parsed: Partial<DailyReport>;
  try {
    parsed = JSON.parse(cleaned) as Partial<DailyReport>;
  } catch (strictErr) {
    // LLMs routinely emit JSON with unescaped quotes inside Chinese
    // strings (e.g. 商务部回应"内卷"). jsonrepair fixes most of these
    // mechanically before we ever surface a failure.
    try {
      const repaired = jsonrepair(cleaned);
      parsed = JSON.parse(repaired) as Partial<DailyReport>;
      console.warn("[pipeline] JSON.parse failed but jsonrepair recovered");
    } catch {
      try {
        const fs = await import("node:fs");
        fs.mkdirSync("logs", { recursive: true });
        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        fs.writeFileSync(`logs/claude-raw-${ts}.txt`, text, "utf8");
        fs.writeFileSync(`logs/claude-cleaned-${ts}.txt`, cleaned, "utf8");
        console.warn(
          `[pipeline] both JSON.parse and jsonrepair failed; raw at logs/claude-raw-${ts}.txt`,
        );
      } catch {
        // best-effort logging
      }
      throw strictErr;
    }
  }
  return {
    hero_headline: parsed.hero_headline ?? "",
    daily_overview: parsed.daily_overview ?? "",
    tech_briefs: parsed.tech_briefs ?? [],
    finance_briefs: parsed.finance_briefs ?? [],
    politics_briefs: parsed.politics_briefs ?? [],
    editor_note: parsed.editor_note ?? "",
    keywords: parsed.keywords ?? [],
  };
}

/** Shape of one item handed to the LLM digest prompt. */
export interface DigestPayloadItem {
  n: number;
  title: string;
  url: string;
  source: string;
  category: Category;
  excerpt: string;
  published: string;
}

/**
 * Build the exact candidate-news payload the digest LLM receives, so both
 * `generateDailyReport` and `scripts/dry-run.ts` share one definition and the
 * dry-run can print / reproduce the precise input that got blocked.
 */
export function buildDigestPayload(articles: ArticleInput[]): DigestPayloadItem[] {
  const grouped: Record<Category, ArticleInput[]> = {
    tech: [],
    finance: [],
    politics: [],
  };
  for (const a of articles) grouped[a.category].push(a);

  const compact = (Object.keys(grouped) as Category[]).flatMap((c) =>
    selectRoundRobin(grouped[c], PER_CATEGORY_LIMIT[c]),
  );

  return compact.map((a, i) => ({
    n: i + 1,
    title: a.title,
    url: a.url,
    source: a.source,
    category: a.category,
    excerpt: (a.excerpt ?? "").slice(0, 200),
    published: a.publishedAt?.toISOString() ?? "",
  }));
}

/**
 * SenseNova (and some providers) reject the request with HTTP 400
 * "Inappropriate input/output rejected for security reasons" when the
 * *content* trips their moderation filter. This is a content-safety block,
 * not a transient/quota/network error — retrying the identical prompt will
 * fail again, so we detect it and handle it specially.
 */
const SECURITY_BLOCK_RE =
  /inappropriate input\/output rejected for security reasons/i;

export function isSecurityBlock(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err ?? "");
  return SECURITY_BLOCK_RE.test(msg);
}

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Strip free-text excerpts (the usual moderation trigger), keep metadata. */
function sanitizePayload(payload: DigestPayloadItem[]): string {
  return JSON.stringify(payload.map((p) => ({ ...p, excerpt: "" })));
}

/** Best-effort dump of the candidate-news payload when a block happens. */
async function dumpSecurityBlockPayload(
  json: string,
  stage: string,
): Promise<void> {
  try {
    const fs = await import("node:fs");
    fs.mkdirSync("logs", { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    fs.writeFileSync(`logs/security-block-${stage}-${ts}.json`, json, "utf8");
    const n = JSON.parse(json).length;
    console.warn(
      `[pipeline] candidate news payload (${n} items) dumped to logs/security-block-${stage}-${ts}.json`,
    );
  } catch {
    // best-effort
  }
}

/**
 * Degraded fallback when the LLM is unavailable or blocked: assemble a report
 * straight from the raw article titles/excerpts so the site still updates
 * (non-blank, correct date) instead of blacking out the whole pipeline.
 */
function buildDegradedReport(items: DigestPayloadItem[]): DailyReport {
  const makeBrief = (a: DigestPayloadItem): BriefItem => ({
    title: a.title,
    url: a.url,
    source: a.source,
    summary: a.excerpt.slice(0, 160) || a.title,
    importance: 5,
  });
  const tech = items.filter((a) => a.category === "tech").map(makeBrief);
  const finance = items.filter((a) => a.category === "finance").map(makeBrief);
  const politics = items
    .filter((a) => a.category === "politics")
    .map(makeBrief);
  const top = items.slice(0, 5);
  const hero = top[0]?.title ?? "每日简报";
  const overview =
    `本期由系统自动降级生成（AI 摘要服务暂不可用，可能是内容安全拦截）。` +
    `以下为当日抓取到的 ${items.length} 条候选要闻标题速览：` +
    top.map((a, i) => `${i + 1}. ${a.title}`).join("；");
  return {
    hero_headline: hero,
    daily_overview: overview,
    tech_briefs: tech.slice(0, 5),
    finance_briefs: finance.slice(0, 5),
    politics_briefs: politics.slice(0, 3),
    editor_note:
      "本期为降级版本：AI 摘要生成失败（疑似内容安全拦截），内容由原始标题/摘要自动拼装，仅供参考。",
    keywords: [...new Set(items.map((a) => a.category))],
  };
}

export async function generateDailyReport(
  articles: ArticleInput[],
): Promise<{ report: DailyReport; tokensUsed: number; degraded: boolean }> {
  const payload = buildDigestPayload(articles);
  const userPayloadJson = JSON.stringify(payload);
  const sanitizedJson = sanitizePayload(payload);

  let report: DailyReport | null = null;
  let lastErr: unknown = null;

  // Attempt 1: normal payload.
  try {
    report = await callOnce(userPayloadJson);
  } catch (err) {
    lastErr = err;
    if (isSecurityBlock(err)) {
      // Content-safety block: strip free-text excerpts (the usual trigger)
      // and retry once with a sanitized payload.
      console.warn(
        `[pipeline] security block on digest (${errMsg(err)}); stripping excerpts and retrying once`,
      );
      await dumpSecurityBlockPayload(userPayloadJson, "primary");
      try {
        report = await callOnce(sanitizedJson);
      } catch (err2) {
        lastErr = err2;
      }
    } else {
      // Legacy behavior: retry once with the same prompt (transient blips).
      console.warn(
        `[pipeline] first digest call failed, retrying: ${errMsg(err)}`,
      );
      try {
        report = await callOnce(userPayloadJson);
      } catch (err2) {
        lastErr = err2;
      }
    }
  }

  if (report) {
    return { report, tokensUsed: 0, degraded: false };
  }

  // All LLM attempts failed — publish a degraded report so the site still
  // updates (non-blank, correct date) instead of going blank.
  console.warn(
    `[pipeline] all LLM digest attempts failed${
      lastErr ? ` (${errMsg(lastErr)})` : ""
    }; publishing degraded report from raw articles`,
  );
  await dumpSecurityBlockPayload(userPayloadJson, "degraded");
  return { report: buildDegradedReport(payload), tokensUsed: 0, degraded: true };
}
