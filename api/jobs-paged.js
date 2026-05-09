const { sliceFromR2, DEFAULT_R2_CSV_URL } = require("../lib/jobs-snapshot-r2.js");

const PER_PAGE_DEFAULT = Math.max(1, Number(process.env.JDP_PAGE_SIZE) || 500);
const PER_PAGE_MAX = 2500;

/** Vercel-style handler: same contract as netlify/functions/jobs-paged.js */
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
  const page = Math.max(1, Number(q.page) || 1);

  const rawPer =
    Number(q.per_page) ||
    Number(q.limit) ||
    PER_PAGE_DEFAULT;
  const perPage = Math.min(PER_PAGE_MAX, Math.max(1, Number.isFinite(rawPer) ? rawPer : PER_PAGE_DEFAULT));

  const offset = (page - 1) * perPage;
  const r2Url = q.r2_url || process.env.JDP_R2_SNAPSHOT_URL || DEFAULT_R2_CSV_URL;

  try {
    const { snapshot, jobs } = await sliceFromR2({
      url: r2Url,
      countryCode,
      offset,
      limit: perPage
    });

    const totalFiltered = Number(snapshot.filtered_total) || 0;
    const totalPages = totalFiltered === 0 ? 0 : Math.ceil(totalFiltered / perPage);

    res.setHeader("Cache-Control", "s-maxage=120, stale-while-revalidate=600");
    return res.status(200).json({
      ok: true,
      pagination: {
        page,
        per_page: perPage,
        offset,
        total_filtered_rows: totalFiltered,
        total_pages: totalPages,
        has_prev: page > 1,
        has_next: totalPages > 0 && page < totalPages,
        shard_source: "live-r2-scan"
      },
      snapshot,
      jobs
    });
  } catch (err) {
    res.setHeader("Cache-Control", "no-store");
    return res.status(502).json({
      ok: false,
      error: "R2 snapshot unavailable",
      detail: err && err.message ? err.message : String(err),
      pagination: {
        page,
        per_page: perPage,
        offset,
        total_filtered_rows: null,
        total_pages: null,
        has_prev: false,
        has_next: false
      },
      snapshot: {
        source: "cloudflare-r2",
        csv_url: r2Url,
        attempted: true
      },
      jobs: []
    });
  }
};
