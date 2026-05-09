const PATHS_TO_SKIP = [
  "/.netlify/",
  "/usage-dashboard",
];

function shouldTrackRequest(request) {
  if (!request) return false;

  const method = String(request.method || "").toUpperCase();
  if (method !== "GET" && method !== "HEAD") return false;

  const url = new URL(request.url);
  return !PATHS_TO_SKIP.some((prefix) => url.pathname.startsWith(prefix));
}

async function forwardCrawlerHit(request) {
  const endpoint = new URL("/.netlify/functions/track-crawler-hit", request.url);
  const clientIp =
    request.headers.get("x-nf-client-connection-ip") ||
    request.headers.get("x-forwarded-for") ||
    request.headers.get("cf-connecting-ip") ||
    "";

  const payload = {
    url: request.url,
    method: request.method,
    user_agent: request.headers.get("user-agent") || "",
    sec_ch_ua_platform: request.headers.get("sec-ch-ua-platform") || "",
    referer: request.headers.get("referer") || "",
    ip_address: clientIp,
    country: request.headers.get("x-nf-geo-country") || "",
  };

  await fetch(endpoint.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export default async (request, context) => {
  if (shouldTrackRequest(request)) {
    context.waitUntil(
      forwardCrawlerHit(request).catch((error) => {
        console.error("crawler-tracker failed:", error?.message || error);
      }),
    );
  }

  return context.next();
};

export const config = {
  path: "/*",
};
