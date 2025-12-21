// netlify/functions/get-jobs.js
//
// Proxies to your listings API:
//   GET  /.netlify/functions/get-jobs?limit=10&industries=Software,Government&country_code=US
//   POST /.netlify/functions/get-jobs   { "limit": 10, "industries": ["Software"], "country_code": "US" }
//
// It calls:
//   https://data.mewannajob.com/listings?limit=10&country_code=US
//
// Then (optionally) filters by industries client-side and de-dupes by job_title.

const LISTINGS_BASE_URL =
  process.env.LISTINGS_BASE_URL || "https://data.mewannajob.com";

function parseIndustriesFromEvent(event) {
  let industries = [];

  // Query string: ?industries=a,b,c
  if (event.queryStringParameters?.industries) {
    industries = event.queryStringParameters.industries
      .split(",")
      .map((part) => decodeURIComponent(part).trim())
      .filter(Boolean);
  }

  // POST body: { industries: [...] }
  if (!industries.length && event.httpMethod === "POST" && event.body) {
    try {
      const body = JSON.parse(event.body);
      if (Array.isArray(body.industries)) {
        industries = body.industries.map((s) => String(s).trim()).filter(Boolean);
      }
    } catch (e) {
      console.warn("Failed to parse JSON body:", e);
    }
  }

  return industries;
}

function parseLimitFromEvent(event) {
  // support ?limit=10 or POST { limit: 10 }
  let limit = 10;

  const qsLimit = event.queryStringParameters?.limit;
  if (qsLimit != null) {
    const n = Number(qsLimit);
    if (Number.isFinite(n)) limit = n;
  } else if (event.httpMethod === "POST" && event.body) {
    try {
      const body = JSON.parse(event.body);
      if (body.limit != null) {
        const n = Number(body.limit);
        if (Number.isFinite(n)) limit = n;
      }
    } catch (_) {}
  }

  // keep it sane
  limit = Math.max(1, Math.min(100, Math.floor(limit)));
  return limit;
}

function parseCountryCodeFromEvent(event) {
  // support ?country_code=US or POST { country_code: "US" }
  let cc = event.queryStringParameters?.country_code;

  if (!cc && event.httpMethod === "POST" && event.body) {
    try {
      const body = JSON.parse(event.body);
      cc = body.country_code;
    } catch (_) {}
  }

  if (!cc) return null;
  cc = String(cc).trim().toUpperCase();
  if (cc.length !== 2) return null;
  return cc;
}

function normalizeIndustries(val) {
  // job_industries might be an array OR a string depending on your DB schema
  if (Array.isArray(val)) {
    return val.map((s) => String(s).trim().toLowerCase()).filter(Boolean);
  }
  if (typeof val === "string") {
    // fallback: split on commas if someone stored "Engineering, Government"
    return val
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);
  }
  return [];
}

function filterByIndustries(items, industries) {
  if (!industries?.length) return items;
  const wanted = industries.map((s) => s.toLowerCase());
  return (items || []).filter((row) => {
    const rowIndustries = normalizeIndustries(row.job_industries);
    // match if any requested industry is present (substring match)
    return wanted.some((w) =>
      rowIndustries.some((ri) => ri.includes(w))
    );
  });
}

function uniqueByJobTitle(items) {
  const seen = new Set();
  const out = [];

  for (const row of items || []) {
    const titleKey = String(row.job_title || "")
      .trim()
      .toLowerCase();
    if (!titleKey) continue;

    if (seen.has(titleKey)) continue;
    seen.add(titleKey);
    out.push(row);
  }
  return out;
}

exports.handler = async (event) => {
  try {
    const industries = parseIndustriesFromEvent(event);
    const limit = parseLimitFromEvent(event);
    const countryCode = parseCountryCodeFromEvent(event);

    // Build upstream URL
    const url = new URL("/listings", LISTINGS_BASE_URL);
    url.searchParams.set("limit", String(limit));
    if (countryCode) url.searchParams.set("country_code", countryCode);

    // If you later add server-side filters (recommended), pass them through:
    // if (industries.length) url.searchParams.set("industries", industries.join(","));

    const resp = await fetch(url.toString(), {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    const text = await resp.text();
    let payload;
    try {
      payload = JSON.parse(text);
    } catch (e) {
      return {
        statusCode: 502,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          error: "Upstream did not return JSON",
          status: resp.status,
          body: text?.slice(0, 500),
        }),
      };
    }

    if (!resp.ok) {
      return {
        statusCode: resp.status,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          error: "Upstream error",
          status: resp.status,
          details: payload,
        }),
      };
    }

    // Your FastAPI returns { count, items }
    const items = Array.isArray(payload) ? payload : payload.items || [];

    // Optional: filter by industries on the Netlify side (since upstream may not support it yet)
    const filtered = filterByIndustries(items, industries);

    // De-dupe by title (same behavior you had before)
    const unique = uniqueByJobTitle(filtered);

    // You asked for /listings?limit=10 — so cap again AFTER filtering/dedupe
    const finalItems = unique.slice(0, limit);

    return {
      statusCode: 200,
      headers: {
        "Content-Type": "application/json",
        // useful for debugging/observability
        "Cache-Control": "no-store",
      },
      body: JSON.stringify({
        upstream: url.toString(),
        count: finalItems.length,
        items: finalItems,
      }),
    };
  } catch (error) {
    console.error("Error fetching jobs:", error);
    return {
      statusCode: 500,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Failed to fetch jobs: " + error.message }),
    };
  }
};

