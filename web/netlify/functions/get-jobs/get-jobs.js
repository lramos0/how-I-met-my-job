// netlify/functions/get-jobs.js
//
// Queries Databricks UC table via Statement Execution API.
// Caches full jobs result in Netlify Blobs for 7 days; only queries DB when cache is missing or expired.
// GET  /.netlify/functions/get-jobs?limit=10&industries=Software,Government&country_code=US
// POST /.netlify/functions/get-jobs { "limit": 10, "industries": ["Software"], "country_code": "US" }

const CACHE_KEY = "jobs-full";
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
const CACHE_FETCH_LIMIT = 500; // rows to fetch and cache when refilling

function parseJsonBody(event) {
  if (event.httpMethod !== "POST" || !event.body) return null;
  try {
    return JSON.parse(event.body);
  } catch {
    return null;
  }
}

function parseIndustriesFromEvent(event) {
  let industries = [];

  if (event.queryStringParameters?.industries) {
    industries = event.queryStringParameters.industries
      .split(",")
      .map((part) => decodeURIComponent(part).trim())
      .filter(Boolean);
  }

  if (!industries.length) {
    const body = parseJsonBody(event);
    if (body && Array.isArray(body.industries)) {
      industries = body.industries.map((s) => String(s).trim()).filter(Boolean);
    }
  }

  return industries;
}

function parseLimitFromEvent(event) {
  let limit = 10;

  const qsLimit = event.queryStringParameters?.limit;
  if (qsLimit != null) {
    const n = Number(qsLimit);
    if (Number.isFinite(n)) limit = n;
  } else {
    const body = parseJsonBody(event);
    if (body?.limit != null) {
      const n = Number(body.limit);
      if (Number.isFinite(n)) limit = n;
    }
  }

  limit = Math.max(1, Math.min(100, Math.floor(limit)));
  return limit;
}

function parseCountryCodeFromEvent(event) {
  let cc = event.queryStringParameters?.country_code;

  if (!cc) {
    const body = parseJsonBody(event);
    cc = body?.country_code;
  }

  if (!cc) return null;
  cc = String(cc).trim().toUpperCase();
  if (cc.length !== 2) return null;
  return cc;
}

function normalizeIndustries(val) {
  if (Array.isArray(val)) {
    return val.map((s) => String(s).trim().toLowerCase()).filter(Boolean);
  }
  if (typeof val === "string") {
    // Could be JSON string '["Software"]' or CSV 'Software, Government'
    const s = val.trim();
    if (!s) return [];
    try {
      const parsed = JSON.parse(s);
      if (Array.isArray(parsed)) {
        return parsed.map((x) => String(x).trim().toLowerCase()).filter(Boolean);
      }
    } catch (_) {}
    return s
      .split(",")
      .map((x) => x.trim().toLowerCase())
      .filter(Boolean);
  }
  return [];
}

function filterByIndustries(items, industries) {
  if (!industries?.length) return items;
  const wanted = industries.map((s) => s.toLowerCase());
  return (items || []).filter((row) => {
    const rowIndustries = normalizeIndustries(row.job_industries);
    return wanted.some((w) => rowIndustries.some((ri) => ri.includes(w)));
  });
}

function uniqueByJobTitle(items) {
  const seen = new Set();
  const out = [];
  for (const row of items || []) {
    const titleKey = String(row.job_title || "").trim().toLowerCase();
    if (!titleKey) continue;
    if (seen.has(titleKey)) continue;
    seen.add(titleKey);
    out.push(row);
  }
  return out;
}

// ---- Databricks SQL Statement Execution API helpers ----

async function dbxExecuteStatement({ host, token, warehouseId, statement }) {
  const url = `${host}/api/2.0/sql/statements`;

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      warehouse_id: warehouseId,
      statement,
      // Make results easy to parse:
      disposition: "INLINE",
      format: "JSON_ARRAY", // returns each row as an array in order of columns
      wait_timeout: "30s",
      on_wait_timeout: "CANCEL",
    }),
  });

  const text = await resp.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    payload = { raw: text };
  }

  if (!resp.ok) {
    return { ok: false, status: resp.status, payload };
  }

  return { ok: true, status: resp.status, payload };
}

function rowsToObjects(payload) {
  // With format JSON_ARRAY, payload.result.data_array is like:
  // [ [col1, col2, ...], [col1, col2, ...] ]
  const manifest = payload?.manifest;
  const schemaCols = manifest?.schema?.columns || [];
  const colNames = schemaCols.map((c) => c.name);

  const data = payload?.result?.data_array || [];
  const out = [];
  for (const row of data) {
    const obj = {};
    for (let i = 0; i < colNames.length; i++) {
      obj[colNames[i]] = row[i];
    }
    out.push(obj);
  }
  return out;
}

function escapeSqlString(s) {
  // Basic SQL string literal escaping
  return String(s).replace(/'/g, "''");
}

function applyFiltersAndSlice(items, industries, limit) {
  const filtered = filterByIndustries(items, industries);
  const unique = uniqueByJobTitle(filtered);
  return unique.slice(0, limit);
}

function successResponse(finalItems, fromCache = false) {
  return {
    statusCode: 200,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": fromCache ? "public, max-age=604800" : "public, max-age=604800", // 7 days
      "X-Jobs-Count": String(finalItems.length),
      "X-Jobs-Cache": fromCache ? "HIT" : "MISS",
    },
    body: JSON.stringify(finalItems),
  };
}

exports.handler = async (event) => {
  try {
    const industries = parseIndustriesFromEvent(event);
    const limit = parseLimitFromEvent(event);
    const countryCode = parseCountryCodeFromEvent(event);

    // ---- Try cache first (Netlify Blobs, 7-day TTL) ----
    let items = null;
    try {
      const { getStore } = await import("@netlify/blobs");
      const store = getStore("get-jobs-cache");
      const raw = await store.get(CACHE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.items) && parsed.cachedAt) {
          const age = Date.now() - parsed.cachedAt;
          if (age < CACHE_TTL_MS) {
            items = parsed.items;
          }
        }
      }
    } catch (blobErr) {
      console.warn("Cache read failed, will query DB:", blobErr.message);
    }

    if (items !== null) {
      const finalItems = applyFiltersAndSlice(items, industries, limit);
      return successResponse(finalItems, true);
    }

    // ---- Cache miss or expired: query Databricks ----
    let host = process.env.DBX_HOST;
    const token = process.env.DBX_TOKEN;
    const warehouseId = process.env.DBX_WAREHOUSE_ID;

    if (!host || !token || !warehouseId) {
      return {
        statusCode: 500,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          error:
            "Missing Databricks config. Set DBX_HOST, DBX_TOKEN, DBX_WAREHOUSE_ID in Netlify env vars.",
        }),
      };
    }

    host = String(host).trim();
    if (!host.startsWith("http://") && !host.startsWith("https://")) {
      host = `https://${host}`;
    }
    host = host.replace(/\/+$/, "");

    const catalog = process.env.DBX_CATALOG || "ml";
    const schema = process.env.DBX_SCHEMA || "job_artifacts";
    const table = process.env.DBX_TABLE || "job_listings";
    const fqtn = `${catalog}.${schema}.${table}`;

    let where = "1=1";
    const statement = `
      SELECT
        id,
        job_title,
        company_name,
        job_location,
        job_seniority_level,
        job_employment_type,
        job_industries,
        job_summary,
        job_base_pay_range,
        job_posted_date,
        competitiveness_score,
        skills,
        certifications,
        industries,
        achievements,
        url
      FROM ${fqtn}
      WHERE ${where}
      ORDER BY job_posted_date DESC
      LIMIT ${CACHE_FETCH_LIMIT}
    `;

    const res = await dbxExecuteStatement({ host, token, warehouseId, statement });

    if (!res.ok) {
      console.error("Databricks SQL error:", res.status, res.payload);
      return {
        statusCode: res.status,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          error: "Databricks SQL query failed",
          status: res.status,
          details: res.payload,
        }),
      };
    }

    items = rowsToObjects(res.payload);

    // Persist to cache for 7 days
    try {
      const { getStore } = await import("@netlify/blobs");
      const store = getStore("get-jobs-cache");
      await store.set(CACHE_KEY, JSON.stringify({ cachedAt: Date.now(), items }));
    } catch (blobErr) {
      console.warn("Cache write failed:", blobErr.message);
    }

    const finalItems = applyFiltersAndSlice(items, industries, limit);
    return successResponse(finalItems, false);
  } catch (error) {
    console.error("Error querying jobs:", error);
    return {
      statusCode: 500,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Failed to fetch jobs: " + error.message }),
    };
  }
};