/**
 * Streams Job Data Pool JSON through Netlify Edge (low buffering; geo-close to users).
 * Compatible with app.js extractJobsPayload() — same response shape as the upstream API.
 *
 * Netlify UI: set JDP_UPSTREAM for Functions scope (not netlify.toml — edge cannot read [build] env).
 */
const DEFAULT_UPSTREAM = "https://api.jobdatapool.com/v1/jobs";
const FORWARD_PARAMS = ["limit", "offset", "page", "country_code"];

export const config = {
  path: "/api/jobs-stream",
};

export default async function handler(request, _context) {
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: { Allow: "GET, OPTIONS" },
    });
  }

  if (request.method !== "GET") {
    return json(405, { error: "Method not allowed" }, { Allow: "GET, OPTIONS" });
  }

  const base = Netlify.env.get("JDP_UPSTREAM") || DEFAULT_UPSTREAM;
  let upstream;
  try {
    upstream = new URL(base);
  } catch {
    upstream = new URL(DEFAULT_UPSTREAM);
  }

  const incoming = new URL(request.url);
  for (const key of FORWARD_PARAMS) {
    const value = incoming.searchParams.get(key);
    if (value != null && value !== "") upstream.searchParams.set(key, value);
  }

  try {
    const upstreamRes = await fetch(upstream.toString(), {
      headers: {
        Accept: "application/json",
        "User-Agent": "TheHiringCafe/1.0 (edge-stream)",
      },
    });

    const headers = new Headers();
    const ct = upstreamRes.headers.get("content-type");
    if (ct) headers.set("Content-Type", ct);
    headers.set("Cache-Control", "public, max-age=300, stale-while-revalidate=600");

    return new Response(upstreamRes.body, {
      status: upstreamRes.status,
      headers,
    });
  } catch (err) {
    const detail = err && err.message ? err.message : String(err);
    return json(502, { error: "Job Data Pool upstream unavailable", detail });
  }
}

function json(status, body, extra = {}) {
  const headers = new Headers({
    "Content-Type": "application/json; charset=utf-8",
    ...extra,
  });
  return new Response(JSON.stringify(body), { status, headers });
}
