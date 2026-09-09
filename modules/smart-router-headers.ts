import {
  AIGatewaySmartRouter,
  ZuploContext,
  ZuploRequest,
} from "@zuplo/runtime";

/**
 * Copies the Smart Router classification onto response headers so callers can
 * show what the gateway decided.
 *
 * Smart Router keeps its result on the request context and does not surface it
 * to the client, so without this the only place the routing decision appears is
 * the Zuplo logs — which is no good for a live demo. The store's
 * shopping-assistant-service reads these headers and passes them through to the
 * chat UI.
 *
 * Runs on every app on the route. When Smart Router did not run (the classifier
 * app itself, for instance) there is no result and the response is returned
 * untouched.
 */
export default async function (
  response: Response,
  _request: ZuploRequest,
  context: ZuploContext,
) {
  const result = AIGatewaySmartRouter.get(context);
  if (!result) {
    return response;
  }

  const headers = new Headers(response.headers);
  headers.set("x-ai-intent", result.profile.intent);
  headers.set("x-ai-complexity", result.profile.complexity);
  headers.set("x-ai-confidence", result.profile.confidence.toFixed(2));
  headers.set("x-ai-routing-applied", String(result.smartRouting.applied));
  headers.set("x-ai-routing-reason", result.smartRouting.reason);
  headers.set("x-ai-classifier-model", result.classifierModel);
  headers.set("x-ai-classify-ms", String(Math.round(result.durationMs)));
  if (result.routing.model) {
    headers.set("x-ai-routed-model", result.routing.model);
  }

  // Browsers can only read these cross-origin if they are allow-listed, and the
  // chat UI is served from a different origin than the gateway. The store's
  // Spin function reads them server-side, but exposing them keeps the headers
  // usable from a browser-based test client too.
  headers.set(
    "access-control-expose-headers",
    [
      "x-ai-intent",
      "x-ai-complexity",
      "x-ai-confidence",
      "x-ai-routing-applied",
      "x-ai-routing-reason",
      "x-ai-classifier-model",
      "x-ai-classify-ms",
      "x-ai-routed-model",
    ].join(", "),
  );

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
