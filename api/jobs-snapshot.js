const { sliceFromR2, DEFAULT_R2_CSV_URL } = require("../lib/jobs-snapshot-r2.js");

const JDP_UPSTREAM = process.env.JDP_UPSTREAM || "https://api.jobdatapool.com/v1/jobs";
const SNAPSHOT_BATCHES = Math.max(1, Number(process.env.JDP_SNAPSHOT_BATCHES) || 4);
const SNAPSHOT_BATCH_SIZE = Math.max(1, Number(process.env.JDP_SNAPSHOT_BATCH_SIZE) || 500);
const USE_API_FALLBACK = process.env.JDP_SNAPSHOT_API_FALLBACK === "true";

function extractRows(payload) {
  if (Array.isArray(payload)) return payload;
  if (payload && Array.isArray(payload.jobs)) return payload.jobs;
  if (payload && Array.isArray(payload.data)) return payload.data;
  return [];
}

module.exports = async function handler(req, res) {
  const method = (req.method || "GET").toUpperCase();
  if (method === "OPTIONS") {
    res.setHeader("Allow", "GET, OPTIONS");
    return res.status(204).end();
  }
  if (method !== "GET") {
    res.setHeader("Allow", "GET, OPTIONS");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const q = req.query || {};
  const countryCode = q.country_code || "US";
  const offset = Number(q.offset) || 0;
  const limit = Number(q.limit) || SNAPSHOT_BATCH_SIZE;
  const r2Url = q.r2_url || process.env.JDP_R2_SNAPSHOT_URL || DEFAULT_R2_CSV_URL;

  try {
    const { snapshot, jobs } = await sliceFromR2({
      url: r2Url,
      countryCode,
      offset,
      limit
    });
    res.setHeader("Cache-Control", "s-maxage=120, stale-while-revalidate=600");
    return res.status(200).json({ ok: true, snapshot, jobs });
  } catch (r2Err) {
    if (!USE_API_FALLBACK) {
      res.setHeader("Cache-Control", "no-store");
      return res.status(502).json({
        ok: false,
        error: "R2 snapshot unavailable",
        detail: r2Err && r2Err.message ? r2Err.message : String(r2Err),
        snapshot: { source: "cloudflare-r2", csv_url: r2Url, attempted: true },
        jobs: []
      });
    }

    const rows = [];
    for (let page = 1; page <= SNAPSHOT_BATCHES; page += 1) {
      const upstream = new URL(JDP_UPSTREAM);
      upstream.searchParams.set("limit", String(SNAPSHOT_BATCH_SIZE));
      upstream.searchParams.set("offset", String((page - 1) * SNAPSHOT_BATCH_SIZE));
      upstream.searchParams.set("page", String(page));
      upstream.searchParams.set("country_code", countryCode);

      try {
        const upstreamRes = await fetch(upstream, {
          headers: { Accept: "application/json", "User-Agent": "TheHiringCafe/1.0" },
          cache: "no-store"
        });
        if (!upstreamRes.ok) break;
        const payload = await upstreamRes.json();
        const batch = extractRows(payload);
        if (!batch.length) break;
        rows.push(...batch);
      } catch {
        break;
      }
    }

    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=900");
    return res.status(200).json({
      ok: true,
      snapshot: {
        source: "jobdatapool-api-fallback",
        country_code: countryCode,
        generated_at: new Date().toISOString(),
        batch_size: SNAPSHOT_BATCH_SIZE,
        batches: SNAPSHOT_BATCHES,
        rows: rows.length,
        r2_error: r2Err && r2Err.message ? r2Err.message : String(r2Err)
      },
      jobs: rows
    });
  }
};
