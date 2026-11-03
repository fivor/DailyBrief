import "./_env";

import { sources } from "../lib/sources/registry";
import { fetchSource } from "../lib/sources/dispatch";
import {
  buildDigestPayload,
  callOnce,
  isSecurityBlock,
  type ArticleInput,
  type DigestPayloadItem,
} from "../lib/ai/pipeline";

// Source-fetch sanity check + digest-payload inspection. With DRY_RUN_DIGEST=1
// it also reproduces the actual LLM digest call (so a 400 security block can be
// reproduced), and with DRY_RUN_LOCATE_TRIGGER=1 it pinpoints which candidate
// item trips the content-safety filter.
async function main() {
  console.log("Fetching from sources…\n");
  const articles: ArticleInput[] = [];

  const enabled = sources.filter((s) => s.enabled !== false);
  for (const source of enabled) {
    try {
      const items = await fetchSource(source);
      console.log(`  ${source.id.padEnd(20)} ${items.length}`);
      articles.push(...items.map((it) => ({ ...it, source: source.name })));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error(`  ${source.id.padEnd(20)} FAILED — ${msg}`);
    }
  }

  console.log(`\nTotal articles: ${articles.length}`);
  console.log("\nTop 10 articles:");
  articles.slice(0, 10).forEach((a, i) => {
    console.log(`  ${i + 1}. [${a.category}] ${a.title}`);
  });

  // Show exactly what the digest LLM would receive. This is the payload a
  // content-safety block rejects, so inspecting it is how we find the trigger.
  const payload = buildDigestPayload(articles);
  console.log(`\nDigest payload: ${payload.length} candidate items.`);
  console.log(
    "Candidate-news JSON (set DAILY_DEBUG_PROMPT=1 to also see the full prompt):",
  );
  console.log(JSON.stringify(payload, null, 2));

  if (process.env.DRY_RUN_DIGEST !== "1") {
    console.log(
      "\n[dry-run] no LLM call made. Set DRY_RUN_DIGEST=1 to reproduce the digest call; add DRY_RUN_LOCATE_TRIGGER=1 to pinpoint the offending item.",
    );
    return;
  }

  console.log("\n[dry-run] reproducing digest call…");
  process.env.DAILY_DEBUG_PROMPT = "1";
  try {
    await callOnce(JSON.stringify(payload));
    console.log("[dry-run] digest succeeded — no block this run.");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (isSecurityBlock(e)) {
      console.warn(`[dry-run] SECURITY BLOCK reproduced: ${msg}`);
      console.warn(
        "[dry-run] payload dumped to logs/security-block-*.json — inspect it to find the trigger item.",
      );
      if (process.env.DRY_RUN_LOCATE_TRIGGER === "1") {
        await locateTrigger(payload);
      }
    } else {
      console.warn(`[dry-run] digest failed (non-security error): ${msg}`);
    }
  }
}

/**
 * Send each candidate item to the LLM alone to find which one trips the
 * content-safety filter. Costs one LLM call per candidate item, so it only
 * runs when DRY_RUN_LOCATE_TRIGGER=1.
 */
async function locateTrigger(payload: DigestPayloadItem[]): Promise<void> {
  console.log(`\n[dry-run] locating trigger across ${payload.length} items…`);
  for (const item of payload) {
    try {
      await callOnce(JSON.stringify([item]));
      console.log(`  ✓ [${item.category}] ${item.title}`);
    } catch (e) {
      if (isSecurityBlock(e)) {
        console.warn(
          `  ✗ TRIGGER: [${item.category}] ${item.title}\n     excerpt: ${item.excerpt.slice(0, 140)}`,
        );
      } else {
        console.log(`  ? [${item.category}] ${item.title} (non-security error)`);
      }
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
