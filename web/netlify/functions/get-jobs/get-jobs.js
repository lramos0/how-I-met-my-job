// netlify/functions/get-jobs.js
//
// Queries Databricks UC table via Statement Execution API.
// GET  /.netlify/functions/get-jobs?limit=10&industries=Software,Government&country_code=US
// POST /.netlify/functions/get-jobs { "limit": 10, "industries": ["Software"], "country_code": "US" }

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

exports.handler = async (event) => {
  try {
    const industries = parseIndustriesFromEvent(event);
    const limit = parseLimitFromEvent(event);
    const countryCode = parseCountryCodeFromEvent(event);

    let host = process.env.DBX_HOST; // allow with or without https
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

    // auto-prefix scheme if user set only hostname
    host = String(host).trim();
    if (!host.startsWith("http://") && !host.startsWith("https://")) {
      host = `https://${host}`;
    }
    // strip trailing slash for clean concatenation
    host = host.replace(/\/+$/, "");

    const catalog = process.env.DBX_CATALOG || "ml";
    const schema = process.env.DBX_SCHEMA || "job_artifacts";
    const table = process.env.DBX_TABLE || "job_listings";
    const fqtn = `${catalog}.${schema}.${table}`;

    // IMPORTANT: Align this SELECT with your actual table columns.
    // Based on what you showed, apply_link/url/country_code are NOT present in the table.
    let where = "1=1";
    if (countryCode) {
      // Only apply this filter if your table actually has country_code.
      // If it doesn't, remove this block or add the column to the table.
      // where += ` AND country_code = '${escapeSqlString(countryCode)}'`;
    }

    // Prefer newest first; if job_posted_date is string, this is best-effort.
    // If you have ingest_utc_date/hour columns, sort by those.
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
      LIMIT ${limit * 5}
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

    const items = rowsToObjects(res.payload);

    // Optional: filter by industries client-side (works whether job_industries is array or string)
    const filtered = filterByIndustries(items, industries);

    // De-dupe by title
    const unique = uniqueByJobTitle(filtered);

    // Final cap
    const finalItems = unique.slice(0, limit);

    return {
      statusCode: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "X-Jobs-Count": String(finalItems.length),
      },
      body: JSON.stringify(finalItems),
    };
  } catch (error) {
    console.error("Error querying jobs:", error);
    return {
      statusCode: 500,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Failed to fetch jobs: " + error.message }),
    };
  }
};