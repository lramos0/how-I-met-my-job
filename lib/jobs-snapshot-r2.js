/**
 * Shared R2 CSV snapshot loader for serverless jobs-snapshot handlers.
 * Default URL matches jobpool-listings-r2 DVC pointer (MeWannaJob / JobDataPool ecosystem).
 */
const DEFAULT_R2_CSV_URL =
  process.env.JDP_R2_SNAPSHOT_URL ||
  "https://pub-e2c96b2fef074ee0809919335319632f.r2.dev/listings-may-2026.csv";

const CACHE_MS = Math.max(30_000, Number(process.env.JDP_R2_CACHE_MS) || 5 * 60 * 1000);

/** @type {{ text: string | null, fetchedAt: number, url: string }} */
let cache = { text: null, fetchedAt: 0, url: "" };

async function fetchCsvTextFromR2(url = DEFAULT_R2_CSV_URL) {
  const now = Date.now();
  if (cache.text && cache.url === url && now - cache.fetchedAt < CACHE_MS) {
    return cache.text;
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
  cache = { text, fetchedAt: now, url };
  return text;
}

function scanCsvWindow(text, { countryCode, offset, limit }) {
  const out = [];
  const wantedCountry = countryCode ? String(countryCode).trim().toUpperCase() : "";
  const wantedOffset = Math.max(0, Number(offset) || 0);
  const wantedLimit = Math.min(2500, Math.max(1, Number(limit) || 500));

  let headers = null;
  let countryIdx = -1;
  let seenMatchingRows = 0;
  let totalCsvRows = 0;

  let row = [];
  let cell = "";
  let q = false;

  const processRow = (currentRow) => {
    const normalized = currentRow.map((v) => String(v ?? "").trim());
    if (!headers) {
      headers = normalized;
      countryIdx = headers.findIndex((h) => h === "country_code");
      return;
    }
    if (!normalized.some((v) => v !== "")) return;
    totalCsvRows += 1;

    const rawCountry = countryIdx >= 0 ? (normalized[countryIdx] || "").toUpperCase() : "";
    const isCountryMatch = !wantedCountry || !rawCountry || rawCountry === wantedCountry;
    if (!isCountryMatch) return;

    if (seenMatchingRows >= wantedOffset && out.length < wantedLimit) {
      const obj = Object.fromEntries(headers.map((h, idx) => [h, normalized[idx] || ""]));
      out.push(obj);
    }
    seenMatchingRows += 1;
  };

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
      processRow(row);
      row = [];
      cell = "";
    } else {
      cell += c;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    processRow(row);
  }

  return {
    rows: out,
    filteredTotal: seenMatchingRows,
    csvRowsTotal: totalCsvRows
  };
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
  const text = await fetchCsvTextFromR2(sourceUrl);
  const o = Math.max(0, Number(offset) || 0);
  const l = Math.min(2500, Math.max(1, Number(limit) || 500));
  const scan = scanCsvWindow(text, { countryCode, offset: o, limit: l });

  return {
    snapshot: {
      source: "cloudflare-r2",
      csv_url: sourceUrl,
      country_code: countryCode || "",
      filtered_total: scan.filteredTotal,
      csv_rows_total: scan.csvRowsTotal,
      generated_at: new Date().toISOString(),
      offset: o,
      limit: l,
      rows: scan.rows.length,
      cache_ttl_ms: CACHE_MS
    },
    jobs: scan.rows
  };
}

module.exports = {
  DEFAULT_R2_CSV_URL,
  fetchCsvTextFromR2,
  sliceFromR2
};
