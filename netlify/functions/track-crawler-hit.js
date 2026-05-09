const { connectBlobsLambda, trackCrawlerRequest } = require("./_shared/crawler-usage");

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

function asString(value) {
  return value == null ? "" : String(value);
}

function toSyntheticRequest(payload) {
  const rawUrl = asString(payload?.url).trim();
  if (!rawUrl) throw new Error("Missing url");

  const method = asString(payload?.method).toUpperCase() === "HEAD" ? "HEAD" : "GET";
  const headers = new Headers();
  headers.set("user-agent", asString(payload?.user_agent));
  headers.set("sec-ch-ua-platform", asString(payload?.sec_ch_ua_platform));
  headers.set("referer", asString(payload?.referer));
  headers.set("x-client-ip", asString(payload?.ip_address));
  headers.set("x-country", asString(payload?.country));

  return new Request(rawUrl, { method, headers });
}

exports.handler = async (event) => {
  connectBlobsLambda(event);

  if (String(event?.httpMethod || "").toUpperCase() !== "POST") {
    return jsonResponse({ error: "Method not allowed." }, 405);
  }

  try {
    const payload = event?.body ? JSON.parse(event.body) : {};
    await trackCrawlerRequest(toSyntheticRequest(payload));
    return {
      statusCode: 204,
      headers: { "cache-control": "no-store" },
      body: "",
    };
  } catch (error) {
    console.error("track-crawler-hit failed:", error?.message || error);
    return jsonResponse({ error: "Failed to track crawler usage." }, 500);
  }
};
