import {
  AIGatewayModelRouting,
  AIGatewaySmartRouter,
  ZuploContext,
  ZuploRequest,
} from "@zuplo/runtime";

/**
 * Smart Router replacement that classifies with a decision model (Laya, or
 * TypeSafe Jev — both speak POST /v1/systemone) instead of asking a chat LLM to
 * write JSON. Measured on our demo prompts: the chat classifier (Qwen3.5-4B)
 * takes ~1.6 s per request through the gateway; Laya answers in ~15 ms on the
 * GPU (plus the network hop), with equal or better accuracy.
 *
 * The complexity is asked as a `score` question (ordered levels), not `choice`:
 * complexity is ordinal, and on 16 JA/EN prompts `score` got 14/16 where a
 * 3-way `choice` got 9-10/16 (it almost never picked "high").
 *
 * The result is written through AIGatewaySmartRouter.set() as well as used for
 * routing, so the existing smart-router-headers-outbound policy — and the store
 * UI's tier / model / classify-time badges — work unchanged. On timeout or any
 * error the request is left unrouted (the caller's model answers) instead of
 * failing: a demo must never break because the classifier did.
 */

type Tier = "low" | "medium" | "high";

interface LayaRouterOptions {
  url: string;
  apiKey: string;
  timeoutMs?: number;
  minConfidence?: number;
  classifierLabel?: string;
  instructions?: string;
  /** Ordered low → medium → high descriptions for the score question. */
  levels?: [string, string, string];
  modelsByComplexity: Record<Tier, string>;
}

const TIERS: Tier[] = ["low", "medium", "high"];
const DEFAULT_INSTRUCTIONS = "How complex is this request? Classify from the prompt alone.";
const DEFAULT_LEVELS: [string, string, string] = [
  "a single simple question answered in one step (availability, price, one fact, greeting)",
  "needs a few steps or combining information (comparison, recommendation with reasons, summarizing a policy)",
  "asks how to design or change the way systems work (architecture, integrating several systems, scaling, data consistency, security or privacy design)",
];

function lastUserText(body: any): string {
  const messages: any[] = Array.isArray(body?.messages) ? body.messages : [];
  const last = [...messages].reverse().find((m) => m?.role === "user");
  if (!last) return "";
  if (typeof last.content === "string") return last.content;
  if (Array.isArray(last.content)) {
    return last.content
      .filter((p: any) => p?.type === "text" && typeof p.text === "string")
      .map((p: any) => p.text)
      .join("\n");
  }
  return "";
}

export default async function layaRouter(
  request: ZuploRequest,
  context: ZuploContext,
  options: LayaRouterOptions,
): Promise<ZuploRequest | Response> {
  const started = Date.now();
  const timeoutMs = options.timeoutMs ?? 2500;
  const minConfidence = options.minConfidence ?? 0.35;
  const classifierLabel = options.classifierLabel ?? "laya/score";

  let body: any;
  try {
    body = await request.clone().json();
  } catch {
    return request; // not JSON — nothing to classify
  }
  const prompt = lastUserText(body).slice(0, 8000);
  if (!prompt) return request;

  const record = (tier: Tier, confidence: number, applied: boolean, reason: any, model?: string) =>
    AIGatewaySmartRouter.set(context, {
      profile: { intent: "", complexity: tier, confidence, reasons: [] },
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      classifierModel: classifierLabel,
      routing: { model },
      durationMs: Date.now() - started,
      promptSource: "user",
      promptLength: prompt.length,
      promptTruncated: false,
      unknownIntent: false,
      smartRouting: { enabled: true, applied, reason, minConfidence },
    } as any);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(options.url, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${options.apiKey}`,
      },
      body: JSON.stringify({
        state: { body: prompt },
        questions: {
          complexity: {
            type: "score",
            instructions: options.instructions ?? DEFAULT_INSTRUCTIONS,
            criteria: options.levels ?? DEFAULT_LEVELS,
          },
        },
      }),
    });
    if (!res.ok) throw new Error(`decision model returned ${res.status}`);
    const data: any = await res.json();
    const answer = data?.answers?.complexity ?? {};
    // Laya returns the level distribution as {"0": p, "1": p, "2": p}; accept an array too.
    const dist = answer.distribution ?? answer.probabilities;
    const probs: number[] = Array.isArray(dist) ? dist.map(Number) : TIERS.map((_, i) => Number(dist?.[String(i)] ?? 0));
    if (probs.length !== 3 || probs.some((p) => !Number.isFinite(p))) {
      throw new Error("unexpected answer shape");
    }
    const idx = probs.indexOf(Math.max(...probs));
    const tier = TIERS[idx];
    const confidence = probs[idx];
    const model = options.modelsByComplexity[tier];

    if (!model) {
      record(tier, confidence, false, "no-model");
    } else if (confidence < minConfidence) {
      record(tier, confidence, false, "low-confidence", model);
    } else {
      await AIGatewayModelRouting.set(context, { completions: model } as any);
      record(tier, confidence, true, "applied", model);
    }
    context.log.info("laya-router", { tier, confidence, model, durationMs: Date.now() - started });
  } catch (err) {
    // Timeout, network, or shape error: leave the request unrouted rather than fail it.
    context.log.warn("laya-router: classification failed, not routing", {
      error: String(err),
      durationMs: Date.now() - started,
    });
    record("medium", 0, false, "internal-error");
  } finally {
    clearTimeout(timer);
  }
  return request;
}
