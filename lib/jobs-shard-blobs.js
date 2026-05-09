const crypto = require("crypto");

/**
 * Netlify Blobs–backed shards: one JSON blob per shard (fixed row count), plus a meta blob.
 * Build still downloads the full CSV once; reads after that fetch only the shard blob (~per page).
 * Shards are stored in Netlify Blobs (site storage), not uploaded as separate files to Cloudflare R2.
 */

function hashSourceUrl(url) {
  return crypto.createHash("sha256").update(String(url)).digest("hex").slice(0, 24);
}

function metaEntryKey(hash, country) {
  return `meta:${hash}:${String(country).toUpperCase()}`;
}

function shardEntryKey(hash, country, shardIndex) {
  return `shard:${hash}:${String(country).toUpperCase()}:${shardIndex}`;
}

/**
 * @param {*} store — store from @netlify/blobs getStore()
 * @param {{ csvText: string, csvUrl: string, countryCode: string, rowsPerShard: number, forEachFilteredCsvRow: Function }} opts
 */
async function buildJobShardsIntoStore(store, { csvText, csvUrl, countryCode, rowsPerShard, forEachFilteredCsvRow }) {
  const hash = hashSourceUrl(csvUrl);
  const cc = (String(countryCode || "").trim().toUpperCase() || "US");
  const rps = Math.min(2500, Math.max(1, Number(rowsPerShard) || 500));

  let totalMatching = 0;
  let shardIdx = 0;
  let buf = [];
  const uploads = [];

  forEachFilteredCsvRow(csvText, cc, (row) => {
    totalMatching += 1;
    buf.push(row);
    if (buf.length >= rps) {
      const myIdx = shardIdx;
      const jobs = buf;
      buf = [];
      shardIdx += 1;
      uploads.push(
        store.setJSON(shardEntryKey(hash, cc, myIdx), {
          jobs,
          shard_index: myIdx,
          rows_in_shard: jobs.length
        })
      );
    }
  });

  if (buf.length) {
    const myIdx = shardIdx;
    const jobs = buf;
    shardIdx += 1;
    uploads.push(
      store.setJSON(shardEntryKey(hash, cc, myIdx), {
        jobs,
        shard_index: myIdx,
        rows_in_shard: jobs.length
      })
    );
  }

  await Promise.all(uploads);

  const shardCount = shardIdx;
  const meta = {
    source: "netlify-blobs",
    csv_url: csvUrl,
    source_hash: hash,
    country_code: cc,
    rows_per_shard: rps,
    total_filtered_rows: totalMatching,
    shard_count: shardCount,
    built_at: new Date().toISOString()
  };

  await store.setJSON(metaEntryKey(hash, cc), meta);
  return meta;
}

/**
 * @param {*} store — store from @netlify/blobs getStore()
 */
async function readJobShardPage(store, { csvUrl, countryCode, page }) {
  const hash = hashSourceUrl(csvUrl);
  const cc = (String(countryCode || "").trim().toUpperCase() || "US");
  const meta = await store.get(metaEntryKey(hash, cc), { type: "json" });
  if (!meta) return null;

  const pageNum = Math.max(1, Number(page) || 1);
  if (pageNum > meta.shard_count) {
    return {
      meta,
      jobs: [],
      shard_index: null,
      empty: true
    };
  }

  const shardIndex = pageNum - 1;
  const shard = await store.get(shardEntryKey(hash, cc, shardIndex), { type: "json" });
  const jobs = shard && Array.isArray(shard.jobs) ? shard.jobs : [];
  return { meta, jobs, shard_index: shardIndex, empty: jobs.length === 0 };
}

module.exports = {
  hashSourceUrl,
  buildJobShardsIntoStore,
  readJobShardPage
};
