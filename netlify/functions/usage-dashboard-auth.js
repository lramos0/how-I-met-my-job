const {
  USAGE_DASHBOARD_PIN_ENV,
  authStatusFromRequest,
  buildAuthClearCookieHeader,
  buildAuthSetCookieHeader,
  isProvidedPinCorrect,
  usageDashboardAuthConfigState,
} = require("./_shared/usage-dashboard-auth");

function jsonResponse(payload, statusCode = 200, extraHeaders = {}) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extraHeaders,
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
  const request = eventToRequest(event);
  const method = String(request.method || "").toUpperCase();
  const config = usageDashboardAuthConfigState();

  if (!config.configured) {
    return jsonResponse({ error: `Missing ${USAGE_DASHBOARD_PIN_ENV} environment variable.` }, 500);
  }

  if (method === "GET") {
    const status = authStatusFromRequest(request);
    return jsonResponse({ authorized: Boolean(status.ok) });
  }

  if (method === "POST") {
    let payload;
    try {
      payload = event?.body ? JSON.parse(event.body) : {};
    } catch (_) {
      payload = {};
    }

    const pin = String(payload?.pin || "").trim();
    if (!isProvidedPinCorrect(pin)) {
      return jsonResponse(
        { error: "Incorrect PIN." },
        401,
        { "set-cookie": buildAuthClearCookieHeader(request) }
      );
    }

    return {
      statusCode: 204,
      headers: {
        "cache-control": "no-store",
        "set-cookie": buildAuthSetCookieHeader(request),
      },
      body: "",
    };
  }

  if (method === "DELETE") {
    return {
      statusCode: 204,
      headers: {
        "cache-control": "no-store",
        "set-cookie": buildAuthClearCookieHeader(request),
      },
      body: "",
    };
  }

  return jsonResponse({ error: "Method not allowed." }, 405);
};
