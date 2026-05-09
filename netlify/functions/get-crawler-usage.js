const { connectBlobsLambda, readCrawlerMetrics } = require("./_shared/crawler-usage");
const { USAGE_DASHBOARD_PIN_ENV, authStatusFromRequest } = require("./_shared/usage-dashboard-auth");

function jsonResponse(payload, statusCode = 200) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
    body: JSON.stringify(payload),
  };
}

function eventHeaderMap(event) {
  const map = {};
  for (const [key, value] of Object.entries(event?.headers || {})) {
    map[String(key).toLowerCase()] = value == null ? "" : String(value);
  }
  return map;
}

function eventToRequest(event) {
  const headerMap = eventHeaderMap(event);
  const proto = headerMap["x-forwarded-proto"] || "https";
  const host = headerMap.host || "localhost";
  const url = event?.rawUrl ? event.rawUrl : `${proto}://${host}${event?.path || "/"}`;

  return {
    url,
    method: String(event?.httpMethod || "GET").toUpperCase(),
    headers: {
      get(name) {
        return headerMap[String(name || "").toLowerCase()] || "";
      },
    },
  };
}

exports.handler = async (event) => {
  connectBlobsLambda(event);

  try {
    const request = eventToRequest(event);
    const authStatus = authStatusFromRequest(request);
    if (!authStatus.ok) {
      if (authStatus.reason === "missing_pin") {
        return jsonResponse({ error: `Missing ${USAGE_DASHBOARD_PIN_ENV} environment variable.` }, 500);
      }
      return jsonResponse({ error: "Unauthorized." }, 401);
    }

    const metrics = await readCrawlerMetrics();
    const byDateRows = Object.entries(metrics.by_date)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([ingestion_date, bucket]) => ({
        ingestion_date,
        global_count: Number(bucket.global_count) || 0,
        crawler_count: Number(bucket.crawler_count) || 0,
        bots: bucket.bots || {},
        devices: bucket.devices || {},
        sdk_devices: bucket.sdk_devices || bucket.devices || {},
        ip_addresses: bucket.ip_addresses || {},
        countries: bucket.countries || {},
        external_referrers: bucket.external_referrers || {},
      }));

    return jsonResponse({
      version: metrics.version,
      created_at: metrics.created_at,
      updated_at: metrics.updated_at,
      totals: metrics.totals,
      bots: metrics.bots,
      devices: metrics.devices,
      external_referrers: metrics.external_referrers || {},
      by_date: byDateRows,
    });
  } catch (error) {
    console.error("get-crawler-usage failed:", error?.message || error);
    return jsonResponse({ error: "Failed to load crawler usage." }, 500);
  }
};
