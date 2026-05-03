// Netlify Function same-origin Job Data Pool proxy.
// Browser calls /api/jobs?...; netlify.toml rewrites it here.
const JDP_UPSTREAM = process.env.JDP_UPSTREAM || "https://api.jobdatapool.com/v1/jobs";

exports.handler = async function handler(event) {
  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 204, headers: { Allow: "GET, OPTIONS" }, body: "" };
  }
  if (event.httpMethod !== "GET") {
    return { statusCode: 405, headers: { Allow: "GET, OPTIONS", "Content-Type": "application/json" }, body: JSON.stringify({ error: "Method not allowed" }) };
  }

  const upstream = new URL(JDP_UPSTREAM);
  const params = event.queryStringParameters || {};
  for (const key of ["limit", "offset", "page", "country_code"]) {
    if (params[key] != null && params[key] !== "") upstream.searchParams.set(key, params[key]);
  }

  try {
    const res = await fetch(upstream.toString(), {
      headers: { Accept: "application/json", "User-Agent": "TheHiringCafe/1.0" },
      cache: "no-store"
    });
    const text = await res.text();
    return {
      statusCode: res.status,
      headers: {
        "Content-Type": res.headers.get("content-type") || "application/json; charset=utf-8",
        "Cache-Control": "public, max-age=300, stale-while-revalidate=600"
      },
      body: text
    };
  } catch (err) {
    return { statusCode: 502, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ error: "Job Data Pool upstream unavailable", detail: err && err.message ? err.message : String(err) }) };
  }
};
