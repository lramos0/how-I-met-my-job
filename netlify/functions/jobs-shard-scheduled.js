const { connectLambda, getStore } = require("@netlify/blobs");
const {
  fetchCsvTextFromR2,
  forEachFilteredCsvRow,
  DEFAULT_R2_CSV_URL
} = require("../../lib/jobs-snapshot-r2");
const { buildJobShardsIntoStore } = require("../../lib/jobs-shard-blobs");

/**
 * Netlify Scheduled Function only — runs on the cron in netlify.toml (not meant as a public API).
 * Rebuilds CSV → shard blobs for GET jobs-paged to consume.
 *
 * Configure: JDP_R2_SNAPSHOT_URL, JDP_SHARD_COUNTRY (default US), JDP_SHARD_ROWS, JDP_BLOB_STORE
 */
exports.handler = async function handler(event) {
  connectLambda(event);

  const countryCode = process.env.JDP_SHARD_COUNTRY || "US";
  const rowsPerShard = Math.min(
    2500,
    Math.max(1, Number(process.env.JDP_SHARD_ROWS) || 500)
  );
  const r2Url = process.env.JDP_R2_SNAPSHOT_URL || DEFAULT_R2_CSV_URL;

  let csvText;
  try {
    csvText = await fetchCsvTextFromR2(r2Url);
  } catch (err) {
    const detail = err && err.message ? err.message : String(err);
    console.error("jobs-shard-scheduled: CSV fetch failed", detail);
    return {
      statusCode: 502,
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ ok: false, error: "Failed to fetch CSV", detail })
    };
  }

  const store = getStore(process.env.JDP_BLOB_STORE || "jdp-listings-shards");

  try {
    const meta = await buildJobShardsIntoStore(store, {
      csvText,
      csvUrl: r2Url,
      countryCode,
      rowsPerShard,
      forEachFilteredCsvRow
    });
    console.log("jobs-shard-scheduled: ok", meta.shard_count, "shards");
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ ok: true, meta })
    };
  } catch (err) {
    const detail = err && err.message ? err.message : String(err);
    console.error("jobs-shard-scheduled: build failed", detail);
    return {
      statusCode: 500,
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ ok: false, error: "Shard build failed", detail })
    };
  }
};
