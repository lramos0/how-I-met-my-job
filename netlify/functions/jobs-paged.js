const { connectLambda, getStore } = require("@netlify/blobs");
const { sliceFromR2, DEFAULT_R2_CSV_URL } = require("../../lib/jobs-snapshot-r2");
const { readJobShardPage } = require("../../lib/jobs-shard-blobs");

const PER_PAGE_DEFAULT = Math.max(1, Number(process.env.JDP_PAGE_SIZE) || 500);
const PER_PAGE_MAX = 2500;

const USE_BLOBS = process.env.JDP_USE_BLOB_SHARDS !== "false";

/**
 * GET: one page of listings.
 * If Netlify Blobs shards exist (see jobs-shard-scheduled), reads only that shard blob.
 * Otherwise scans R2 CSV via sliceFromR2 (full file in memory).
 */
exports.handler = async function handler(event) {
  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 204, headers: { Allow: "GET, OPTIONS" }, body: "" };
  }
  if (event.httpMethod !== "GET") {
    return json(405, { error: "Method not allowed" }, { Allow: "GET, OPTIONS" });
  }

  const q = event.queryStringParameters || {};
  const countryCode = q.country_code || "US";
  const page = Math.max(1, Number(q.page) || 1);

  const rawPer =
    Number(q.per_page) ||
    Number(q.limit) ||
    PER_PAGE_DEFAULT;
  const requestedPerPage = Math.min(PER_PAGE_MAX, Math.max(1, Number.isFinite(rawPer) ? rawPer : PER_PAGE_DEFAULT));

  const r2Url = q.r2_url || process.env.JDP_R2_SNAPSHOT_URL || DEFAULT_R2_CSV_URL;

  if (USE_BLOBS) {
    try {
      connectLambda(event);
      const store = getStore(process.env.JDP_BLOB_STORE || "jdp-listings-shards");
      const blobResult = await readJobShardPage(store, {
        csvUrl: r2Url,
        countryCode,
        page
      });

      if (blobResult && blobResult.meta) {
        const meta = blobResult.meta;
        const perPage = meta.rows_per_shard;
        const offset = (page - 1) * perPage;
        const totalFiltered = meta.total_filtered_rows;
        const totalPages = meta.shard_count;

        const pagination = {
          page,
          per_page: perPage,
          requested_per_page: requestedPerPage,
          per_page_matches_shard: requestedPerPage === perPage,
          offset,
          total_filtered_rows: totalFiltered,
          total_pages: totalPages,
          has_prev: page > 1,
          has_next: totalPages > 0 && page < totalPages,
          shard_source: "netlify-blobs",
          shard_index: blobResult.shard_index
        };

        const snapshot = {
          source: "netlify-blobs-shard",
          csv_url: meta.csv_url,
          country_code: meta.country_code,
          filtered_total: totalFiltered,
          rows_per_shard: meta.rows_per_shard,
          shard_count: meta.shard_count,
          built_at: meta.built_at,
          offset,
          limit: perPage,
          rows: blobResult.jobs ? blobResult.jobs.length : 0
        };

        return json(
          200,
          {
            ok: true,
            pagination,
            snapshot,
            jobs: blobResult.jobs || []
          },
          { "Cache-Control": "public, max-age=120, stale-while-revalidate=600" }
        );
      }
    } catch (blobErr) {
      if (process.env.JDP_BLOB_SHARDS_STRICT === "true") {
        const detail = blobErr && blobErr.message ? blobErr.message : String(blobErr);
        return json(502, { ok: false, error: "Blob shards required but unavailable", detail }, { "Cache-Control": "no-store" });
      }
    }
  }

  const offset = (page - 1) * requestedPerPage;

  try {
    const { snapshot, jobs } = await sliceFromR2({
      url: r2Url,
      countryCode,
      offset,
      limit: requestedPerPage
    });

    const totalFiltered = Number(snapshot.filtered_total) || 0;
    const totalPages = totalFiltered === 0 ? 0 : Math.ceil(totalFiltered / requestedPerPage);

    const pagination = {
      page,
      per_page: requestedPerPage,
      offset,
      total_filtered_rows: totalFiltered,
      total_pages: totalPages,
      has_prev: page > 1,
      has_next: totalPages > 0 && page < totalPages,
      shard_source: "live-r2-scan"
    };

    return json(
      200,
      {
        ok: true,
        pagination,
        snapshot,
        jobs
      },
      { "Cache-Control": "public, max-age=120, stale-while-revalidate=600" }
    );
  } catch (err) {
    return json(
      502,
      {
        ok: false,
        error: "R2 snapshot unavailable",
        detail: err && err.message ? err.message : String(err),
        pagination: {
          page,
          per_page: requestedPerPage,
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
      },
      { "Cache-Control": "no-store" }
    );
  }
};

function json(statusCode, body, headers = {}) {
  return {
    statusCode,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...headers
    },
    body: JSON.stringify(body)
  };
}
