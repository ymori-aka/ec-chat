# laya-gateway

`laya_gateway.py` wraps [laya-serve](https://github.com/NandhaKishorM/laya) so that one process answers two routes:

- `/v1/systemone` (native laya-serve) — used by the `laya-router-inbound` policy (`modules/laya-router.ts`)
- `/v1/chat/completions` (added here) — lets Zuplo's Smart Router use Laya as its `classifierModel`

The Smart Router calls its classifier as a chat-completions model and reads `choices[0].message.content` as `{intent, complexity, confidence, reasons}`. This shim maps that call onto Laya's `score` / `choice` questions.

## Notes

- Intent ids come from the request's `response_format` `intent.enum`; their descriptions come from the `- id: description` lines in the system prompt.
- Complexity levels match `modules/laya-router.ts`, so both paths classify identically.
- Laya's top probability over 3 tiers was only ~0.4-0.54 in testing, so `minConfidenceForRouting` needs to be lowered (0.35) for the Smart Router path.
- `usage` in the response is always 0: the decision call is not token-metered.
- Auth: set `LAYA_API_KEY` to require `Authorization: Bearer <key>`.
- Tracing: set `OTEL_EXPORTER_OTLP_ENDPOINT` to emit spans (`laya.predict`, plus `ai_gateway.*` attributes).

## Run

```bash
LAYA_DEVICE=cuda LAYA_MODELS=english,multilingual LAYA_PORT=8010 python laya_gateway.py
```
