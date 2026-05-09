/**
 * Streams the public R2 CSV snapshot through the edge without parsing it on the server.
 * Use for bulk distribution, tooling, or future clients that parse CSV incrementally.
 * Default URL matches lib/jobs-snapshot-r2.js when env is unset.
 *
 * Netlify UI: set JDP_R2_SNAPSHOT_URL (Functions scope) to override the default host/path.
 */
const DEFAULT_R2_CSV_URL =
  "https://pub-e2c96b2fef074ee0809919335319632f.r2.dev/listings-may-2026.csv";

export const config = {
  path: "/api/jobs-csv-stream",
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

  const incoming = new URL(request.url);
  const qUrl = incoming.searchParams.get("r2_url") || "";
  const envUrl = Netlify.env.get("JDP_R2_SNAPSHOT_URL") || "";
  const target = qUrl || envUrl || DEFAULT_R2_CSV_URL;

  let fetchUrl;
  try {
    fetchUrl = new URL(target);
  } catch {
    return json(400, { error: "Invalid r2_url" });
  }

  try {
    const upstreamRes = await fetch(fetchUrl.toString(), {
      headers: {
        Accept: "text/csv, text/plain, */*",
        "User-Agent": "TheHiringCafe/1.0 (edge-csv-stream)",
      },
    });

    const headers = new Headers();
    const ct = upstreamRes.headers.get("content-type");
    headers.set("Content-Type", ct || "text/csv; charset=utf-8");
    headers.set("Cache-Control", "public, max-age=120, stale-while-revalidate=600");

    return new Response(upstreamRes.body, {
      status: upstreamRes.status,
      headers,
    });
  } catch (err) {
    const detail = err && err.message ? err.message : String(err);
    return json(502, { error: "Snapshot CSV upstream unavailable", detail });
  }
}

function json(status, body, extra = {}) {
  const headers = new Headers({
    "Content-Type": "application/json; charset=utf-8",
    ...extra,
  });
  return new Response(JSON.stringify(body), { status, headers });
}
