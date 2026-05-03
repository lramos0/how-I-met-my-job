// Same-origin Job Data Pool proxy for Vercel-style /api deployments.
// Your browser calls /api/jobs?...; this server function calls Job Data Pool.
const JDP_UPSTREAM = process.env.JDP_UPSTREAM || "https://api.jobdatapool.com/v1/jobs";

export default async function handler(req, res) {
  const method = (req.method || "GET").toUpperCase();

  if (method === "OPTIONS") {
    res.setHeader("Allow", "GET, OPTIONS");
    return res.status(204).end();
  }

  if (method !== "GET") {
    res.setHeader("Allow", "GET, OPTIONS");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const url = new URL(req.url, "http://localhost");
  const upstream = new URL(JDP_UPSTREAM);

  for (const key of ["limit", "offset", "page", "country_code"]) {
    const value = url.searchParams.get(key);
    if (value != null && value !== "") upstream.searchParams.set(key, value);
  }

  try {
    const upstreamRes = await fetch(upstream, {
      headers: { Accept: "application/json", "User-Agent": "TheHiringCafe/1.0" },
      cache: "no-store"
    });

    const text = await upstreamRes.text();
    res.setHeader("Content-Type", upstreamRes.headers.get("content-type") || "application/json; charset=utf-8");
    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=600");
    return res.status(upstreamRes.status).send(text);
  } catch (err) {
    return res.status(502).json({ error: "Job Data Pool upstream unavailable", detail: err?.message || String(err) });
  }
}
