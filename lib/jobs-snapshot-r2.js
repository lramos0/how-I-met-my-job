/**
 * Shared R2 CSV snapshot loader for serverless jobs-snapshot handlers.
 * Default URL matches jobpool-listings-r2 DVC pointer (MeWannaJob / JobDataPool ecosystem).
 */
const DEFAULT_R2_CSV_URL =
  process.env.JDP_R2_SNAPSHOT_URL ||
  "https://pub-e2c96b2fef074ee0809919335319632f.r2.dev/listings-may-2026.csv";

const CACHE_MS = Math.max(30_000, Number(process.env.JDP_R2_CACHE_MS) || 5 * 60 * 1000);

/** @type {{ rows: object[] | null, fetchedAt: number, url: string }} */
let cache = { rows: null, fetchedAt: 0, url: "" };

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let q = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    const n = text[i + 1];
    if (c === '"') {
      if (q && n === '"') {
        cell += '"';
        i += 1;
      } else q = !q;
    } else if (c === "," && !q) {
      row.push(cell);
      cell = "";
    } else if ((c === "\n" || c === "\r") && !q) {
      if (c === "\r" && n === "\n") i += 1;
      row.push(cell);
      if (row.some((x) => String(x).trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else cell += c;
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  if (!rows.length) return [];
  const headers = rows.shift().map((h) => String(h).trim());
  return rows.map((r) =>
    Object.fromEntries(headers.map((h, idx) => [h, String(r[idx] ?? "").trim()]))
  );
}

async function fetchRowsFromR2(url = DEFAULT_R2_CSV_URL) {
  const now = Date.now();
  if (cache.rows && cache.url === url && now - cache.fetchedAt < CACHE_MS) {
    return cache.rows;
  }

  const res = await fetch(url, {
    headers: {
      Accept: "text/csv, text/plain, */*",
      "User-Agent": "JobDataPool-jobs-snapshot/1.0 (R2 CSV)"
    },
    cache: "no-store"
  });
  if (!res.ok) {
    throw new Error(`R2 snapshot GET ${url} returned ${res.status}`);
  }

  const text = await res.text();
  const rows = parseCsv(text);
  cache = { rows, fetchedAt: now, url };
  return rows;
}

function filterCountry(rows, countryCode) {
  if (!countryCode) return rows;
  const cc = String(countryCode).toUpperCase();
  return rows.filter((r) => {
    const raw = r.country_code != null ? String(r.country_code).trim().toUpperCase() : "";
    if (!raw) return true;
    return raw === cc;
  });
}

/**
 * Returns a page of job objects (CSV-shaped) from R2 plus snapshot metadata.
 */
async function sliceFromR2({
  url,
  countryCode,
  offset = 0,
  limit = 500
}) {
  const sourceUrl = url || DEFAULT_R2_CSV_URL;
  const all = await fetchRowsFromR2(sourceUrl);
  const filtered = filterCountry(all, countryCode);
  const o = Math.max(0, Number(offset) || 0);
  const l = Math.min(2500, Math.max(1, Number(limit) || 500));
  const page = filtered.slice(o, o + l);

  return {
    snapshot: {
      source: "cloudflare-r2",
      csv_url: sourceUrl,
      country_code: countryCode || "",
      filtered_total: filtered.length,
      csv_rows_total: all.length,
      generated_at: new Date().toISOString(),
      offset: o,
      limit: l,
      rows: page.length,
      cache_ttl_ms: CACHE_MS
    },
    jobs: page
  };
}

module.exports = {
  DEFAULT_R2_CSV_URL,
  parseCsv,
  fetchRowsFromR2,
  sliceFromR2
};
