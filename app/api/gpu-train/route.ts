/**
 * POST /api/gpu-train
 *
 * Proxies training requests to Modal GPU endpoint.
 * Explicitly pipes SSE chunks to avoid framework buffering.
 *
 * Env var: MODAL_ENDPOINT_URL
 */

import { NextRequest } from "next/server";

const MODAL_ENDPOINT_URL = process.env.MODAL_ENDPOINT_URL ?? "";

export async function POST(req: NextRequest) {
  if (!MODAL_ENDPOINT_URL) {
    return Response.json(
      { error: "GPU training not configured (missing MODAL_ENDPOINT_URL)" },
      { status: 503 },
    );
  }

  const body = await req.json();

  const res = await fetch(`${MODAL_ENDPOINT_URL}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    return Response.json(
      { error: `GPU endpoint error: ${res.status} ${text}` },
      { status: 502 },
    );
  }

  // Explicitly pipe chunks — don't rely on pass-through which can buffer
  const upstream = res.body;
  if (!upstream) {
    return Response.json({ error: "No upstream body" }, { status: 502 });
  }

  const stream = new ReadableStream({
    async start(controller) {
      const reader = upstream.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          controller.enqueue(value);
        }
      } catch (err) {
        console.error("[gpu-train] stream pipe error:", err);
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
