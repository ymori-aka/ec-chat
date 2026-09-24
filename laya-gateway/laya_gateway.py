"""laya-serve plus an OpenAI-compatible classifier endpoint for Zuplo Smart Router.

Smart Router calls its classifier as a chat-completions model and reads
choices[0].message.content as {intent, complexity, confidence, reasons}. This
adds that route to laya-serve's own app, so the same process (and GPU model)
answers both /v1/systemone (laya-router-inbound) and /v1/chat/completions
(Smart Router with classifierModel pointed here).

Complexity is asked as a `score` question with the same levels as
modules/laya-router.ts in ec-chat, so both paths classify identically. The
intent ids come from the request's response_format schema (intent.enum) and
their descriptions from the "- id: description" lines in the system prompt.
"""
import asyncio
import hmac
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import Header, HTTPException, Request
from opentelemetry import trace

from laya.serve import _resolve_port, build_router, create_app

TIERS = ["low", "medium", "high"]
COMPLEXITY_INSTRUCTIONS = "How complex is this request? Classify from the prompt alone."
COMPLEXITY_LEVELS = [
    "a single simple question answered in one step (availability, price, one fact, greeting)",
    "needs a few steps or combining information (comparison, recommendation with reasons, summarizing a policy)",
    "asks how to design or change the way systems work (architecture, integrating several systems, scaling, data consistency, security or privacy design)",
]

router = build_router()
app = create_app(router)

# Tracing is on when OTEL_EXPORTER_OTLP_ENDPOINT is set (the systemd unit points
# it at the host's local collector, which forwards to Tokyo). Both endpoints get
# a server span; /health is excluded so the NodeBalancer's checks stay out.
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "laya")}))
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
tracer = trace.get_tracer("laya_gateway")
api_key = os.environ.get("LAYA_API_KEY") or None
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-chat")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _intents(body: dict, system: str) -> dict:
    schema = (((body.get("response_format") or {}).get("json_schema") or {}).get("schema") or {})
    ids = (((schema.get("properties") or {}).get("intent") or {}).get("enum")) or []
    out = {}
    for iid in ids:
        m = re.search(r"^- " + re.escape(iid) + r": (.+)$", system, re.M)
        out[iid] = m.group(1).strip() if m else iid
    return out


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None)):
    if api_key is not None and not hmac.compare_digest(authorization or "", "Bearer " + api_key):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="request body must be valid JSON")
    messages = body.get("messages") or []
    system = "\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    user = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
    if not user:
        raise HTTPException(status_code=400, detail="no user message")
    intents = _intents(body, system)

    questions = {
        "complexity": {"type": "score", "instructions": COMPLEXITY_INSTRUCTIONS, "criteria": COMPLEXITY_LEVELS},
    }
    if len(intents) > 1:
        questions["intent"] = {
            "type": "choice",
            "instructions": "Which intent best describes this request?",
            "criteria": intents,
        }

    started = time.time()
    loop = asyncio.get_running_loop()
    with tracer.start_as_current_span("laya.predict") as span:
        span.set_attribute("laya.questions", ",".join(questions))
        span.set_attribute("laya.prompt_chars", len(user))
        try:
            result = await loop.run_in_executor(pool, lambda: router.predict({"body": user}, questions))
        except Exception as e:  # noqa: BLE001 -- same policy as laya-serve: no internals to clients
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise HTTPException(status_code=500, detail="inference failed")
        span.set_attribute("laya.model", str(result.get("model", "")))
    answers = result.get("answers", {})

    probs = answers.get("complexity", {}).get("probabilities", {})
    p = [float(probs.get(str(i), 0)) for i in range(3)]
    idx = max(range(3), key=lambda i: p[i])
    if len(intents) > 1:
        intent = answers.get("intent", {}).get("choice") or next(iter(intents))
    else:
        intent = next(iter(intents), "other")

    span = trace.get_current_span()
    span.set_attribute("ai_gateway.complexity", TIERS[idx])
    span.set_attribute("ai_gateway.confidence", round(p[idx], 4))
    span.set_attribute("ai_gateway.intent", intent)
    profile = {
        "intent": intent,
        "complexity": TIERS[idx],
        "confidence": round(p[idx], 4),
        "reasons": [f"laya score {answers.get('complexity', {}).get('score', 0):.2f}"],
    }
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(started),
        "model": body.get("model") or "laya-router",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps(profile)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("LAYA_HOST", "0.0.0.0"), port=_resolve_port(),
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))
