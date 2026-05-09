const STORE_NAME = "crawler-usage";
const SUMMARY_KEY = "summary-v1";
const RETENTION_DAYS = 30;
const MAX_UPDATE_ATTEMPTS = 12;

const KNOWN_BOTS = [
  { key: "googlebot", label: "Googlebot", pattern: /googlebot/i },
  { key: "oai-searchbot", label: "OAI-SearchBot", pattern: /oai-searchbot/i },
  { key: "facebookexternalhit", label: "facebookexternalhit", pattern: /facebookexternalhit/i },
  { key: "bingbot", label: "Bingbot", pattern: /bingbot/i },
  { key: "applebot", label: "Applebot", pattern: /applebot/i },
  { key: "duckduckbot", label: "DuckDuckBot", pattern: /duckduckbot/i },
  { key: "baiduspider", label: "Baiduspider", pattern: /baiduspider/i },
  { key: "yandexbot", label: "YandexBot", pattern: /yandex(bot)?/i },
  { key: "slurp", label: "Yahoo! Slurp", pattern: /\bslurp\b/i },
];

const GENERIC_BOT_PATTERN = /(bot|crawler|spider|slurp|archiver|fetcher|preview)/i;
const PATHS_TO_SKIP = [
  "/.netlify/",
];

let getStore = null;
let connectLambda = null;
try {
  ({ getStore, connectLambda } = require("@netlify/blobs"));
} catch (_) {
  getStore = null;
  connectLambda = null;
}

function emptyMetrics(nowIso = new Date().toISOString()) {
  return {
    version: 1,
    created_at: nowIso,
    updated_at: nowIso,
    totals: {
      global_count: 0,
      crawler_count: 0,
      external_referrer_count: 0,
    },
    bots: {},
    devices: {},
    external_referrers: {},
    by_date: {},
  };
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function normalizeBucket(bucket) {
  const current = asObject(bucket);
  const legacyDevices = asObject(current.devices);
  const sdkDevices = asObject(current.sdk_devices);
  return {
    global_count: asNumber(current.global_count),
    crawler_count: asNumber(current.crawler_count),
    bots: asObject(current.bots),
    devices: legacyDevices,
    sdk_devices: Object.keys(sdkDevices).length ? sdkDevices : { ...legacyDevices },
    ip_addresses: asObject(current.ip_addresses),
    countries: asObject(current.countries),
    external_referrers: asObject(current.external_referrers),
  };
}

function normalizeMetrics(raw) {
  const nowIso = new Date().toISOString();
  if (!raw || typeof raw !== "object") return emptyMetrics(nowIso);

  const normalized = {
    version: 1,
    created_at: typeof raw.created_at === "string" ? raw.created_at : nowIso,
    updated_at: typeof raw.updated_at === "string" ? raw.updated_at : nowIso,
    totals: {
      global_count: asNumber(raw?.totals?.global_count),
      crawler_count: asNumber(raw?.totals?.crawler_count),
      external_referrer_count: asNumber(raw?.totals?.external_referrer_count),
    },
    bots: asObject(raw.bots),
    devices: {},
    external_referrers: asObject(raw.external_referrers),
    by_date: {},
  };

  for (const [device, bucket] of Object.entries(asObject(raw.devices))) {
    normalized.devices[device] = normalizeBucket(bucket);
  }

  for (const [dateKey, bucket] of Object.entries(asObject(raw.by_date))) {
    normalized.by_date[dateKey] = normalizeBucket(bucket);
  }

  return normalized;
}

function pruneOldDates(byDate) {
  const cutoff = new Date();
  cutoff.setUTCHours(0, 0, 0, 0);
  cutoff.setUTCDate(cutoff.getUTCDate() - (RETENTION_DAYS - 1));
  const cutoffEpoch = cutoff.getTime();

  for (const key of Object.keys(byDate)) {
    const parsed = new Date(`${key}T00:00:00Z`);
    if (Number.isNaN(parsed.getTime())) continue;
    if (parsed.getTime() < cutoffEpoch) delete byDate[key];
  }
}

function incrementMap(map, key, amount = 1) {
  map[key] = asNumber(map[key]) + amount;
}

function cleanDeviceName(value) {
  const normalized = String(value || "").trim().replace(/\s+/g, " ");
  return normalized ? normalized.slice(0, 120) : "unknown";
}

function inferDeviceFromUserAgent(userAgentLower) {
  if (/\b(iphone|android.+mobile|mobile)\b/i.test(userAgentLower)) return "mobile";
  if (/\b(ipad|tablet)\b/i.test(userAgentLower)) return "tablet";
  if (/\b(macintosh|mac os x|windows nt|x11|linux)\b/i.test(userAgentLower)) return "desktop";
  return "unknown";
}

function extractSdkDevice(userAgent, secChUaPlatform = "") {
  const cleanedPlatform = cleanDeviceName(String(secChUaPlatform || "").replaceAll('"', ""));
  if (cleanedPlatform !== "unknown") return cleanedPlatform;

  const ua = String(userAgent || "");
  const parenMatch = ua.match(/\(([^)]+)\)/);
  if (parenMatch?.[1]) {
    const cleaned = cleanDeviceName(parenMatch[1]);
    if (cleaned !== "unknown") return cleaned;
  }

  return inferDeviceFromUserAgent(ua.toLowerCase());
}

function detectCrawler(userAgent = "") {
  const ua = String(userAgent);
  for (const bot of KNOWN_BOTS) {
    if (bot.pattern.test(ua)) return { key: bot.key, label: bot.label };
  }
  if (GENERIC_BOT_PATTERN.test(ua)) return { key: "other-crawler", label: "Other crawler" };
  return null;
}

function cleanIpAddress(value) {
  const raw = String(value || "").trim();
  if (!raw) return "unknown";
  const first = raw.split(",")[0]?.trim();
  return first ? first.slice(0, 96) : "unknown";
}

function extractClientIpAddress(request) {
  const candidates = [
    request?.headers?.get("x-client-ip"),
    request?.headers?.get("x-nf-client-connection-ip"),
    request?.headers?.get("x-forwarded-for"),
    request?.headers?.get("cf-connecting-ip"),
    request?.headers?.get("client-ip"),
    request?.headers?.get("x-real-ip"),
  ];

  for (const candidate of candidates) {
    const cleaned = cleanIpAddress(candidate);
    if (cleaned !== "unknown") return cleaned;
  }

  return "unknown";
}

function requestHost(request) {
  try {
    return new URL(request.url).hostname.toLowerCase();
  } catch (_) {
    return "";
  }
}

function extractExternalReferrerHost(referrerHeader = "", request) {
  const raw = String(referrerHeader || "").trim();
  if (!raw) return null;

  let host;
  try {
    host = new URL(raw).hostname.toLowerCase();
  } catch (_) {
    return null;
  }

  const ownHost = requestHost(request);
  if (!host || (ownHost && (host === ownHost || host.endsWith(`.${ownHost}`)))) return null;
  return host;
}

function resolveCountryCode(request) {
  const candidates = [
    request?.headers?.get("x-nf-geo-country"),
    request?.headers?.get("x-country"),
    request?.headers?.get("cf-ipcountry"),
    request?.headers?.get("x-vercel-ip-country"),
  ];

  for (const raw of candidates) {
    const code = String(raw || "").trim().toUpperCase();
    if (/^[A-Z]{2}$/.test(code)) return code;
  }

  return "ZZ";
}

function shouldTrackRequest(request) {
  if (!request) return false;
  const method = String(request.method || "").toUpperCase();
  if (method !== "GET" && method !== "HEAD") return false;
  const url = new URL(request.url);
  return !PATHS_TO_SKIP.some((prefix) => url.pathname.startsWith(prefix));
}

function applyRequestToMetrics(metrics, request, now = new Date()) {
  const nowIso = now.toISOString();
  const dateKey = nowIso.slice(0, 10);
  const userAgent = request.headers.get("user-agent") || "";
  const platform = request.headers.get("sec-ch-ua-platform") || "";
  const device = extractSdkDevice(userAgent, platform);
  const ipAddress = extractClientIpAddress(request);
  const crawler = detectCrawler(userAgent);
  const countryCode = resolveCountryCode(request);
  const externalReferrer = extractExternalReferrerHost(request.headers.get("referer") || "", request);

  metrics.updated_at = nowIso;
  incrementMap(metrics.totals, "global_count", 1);

  if (!metrics.devices[device]) metrics.devices[device] = normalizeBucket({});
  incrementMap(metrics.devices[device], "global_count", 1);

  if (!metrics.by_date[dateKey]) metrics.by_date[dateKey] = normalizeBucket({});
  incrementMap(metrics.by_date[dateKey], "global_count", 1);
  incrementMap(metrics.by_date[dateKey].sdk_devices, device, 1);
  incrementMap(metrics.by_date[dateKey].ip_addresses, ipAddress, 1);
  incrementMap(metrics.by_date[dateKey].countries, countryCode, 1);

  if (externalReferrer) {
    incrementMap(metrics.totals, "external_referrer_count", 1);
    incrementMap(metrics.external_referrers, externalReferrer, 1);
    incrementMap(metrics.by_date[dateKey].external_referrers, externalReferrer, 1);
  }

  if (crawler) {
    incrementMap(metrics.totals, "crawler_count", 1);
    incrementMap(metrics.bots, crawler.key, 1);
    incrementMap(metrics.devices[device], "crawler_count", 1);
    incrementMap(metrics.devices[device].bots, crawler.key, 1);
    incrementMap(metrics.by_date[dateKey], "crawler_count", 1);
    incrementMap(metrics.by_date[dateKey].bots, crawler.key, 1);
    incrementMap(metrics.by_date[dateKey].devices, device, 1);
  }

  pruneOldDates(metrics.by_date);
}

function nextBackoffMs(attempt) {
  return 5 + attempt * 10;
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function connectBlobsLambda(event) {
  if (typeof connectLambda !== "function") return;
  try {
    connectLambda(event);
  } catch (_) {
    // Best effort; callers can still return a graceful fallback.
  }
}

async function getCrawlerUsageStore() {
  if (typeof getStore !== "function") return null;
  try {
    return getStore(STORE_NAME);
  } catch (error) {
    console.warn("crawler-usage: @netlify/blobs unavailable:", error?.message || error);
    return null;
  }
}

function isStrongConsistencyUnsupported(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return message.includes("uncachededgeurl") || message.includes("strong consistency");
}

async function readSummaryWithMetadata(store) {
  try {
    return await store.getWithMetadata(SUMMARY_KEY, {
      consistency: "strong",
      type: "json",
    });
  } catch (error) {
    if (!isStrongConsistencyUnsupported(error)) throw error;
    return store.getWithMetadata(SUMMARY_KEY, { type: "json" });
  }
}

async function readSummary(store) {
  try {
    return await store.get(SUMMARY_KEY, {
      consistency: "strong",
      type: "json",
    });
  } catch (error) {
    if (!isStrongConsistencyUnsupported(error)) throw error;
    return store.get(SUMMARY_KEY, { type: "json" });
  }
}

async function trackCrawlerRequest(request) {
  const store = await getCrawlerUsageStore();
  if (!store) {
    const fallback = emptyMetrics();
    applyRequestToMetrics(fallback, request);
    return fallback;
  }

  for (let attempt = 0; attempt < MAX_UPDATE_ATTEMPTS; attempt += 1) {
    const current = await readSummaryWithMetadata(store);
    const metrics = normalizeMetrics(current?.data);
    applyRequestToMetrics(metrics, request);

    const writeOptions = current?.etag ? { onlyIfMatch: current.etag } : { onlyIfNew: true };
    const result = await store.set(SUMMARY_KEY, JSON.stringify(metrics), writeOptions);
    if (result?.modified) return metrics;

    await sleep(nextBackoffMs(attempt));
  }

  const current = await readSummary(store);
  const metrics = normalizeMetrics(current);
  applyRequestToMetrics(metrics, request);
  await store.set(SUMMARY_KEY, JSON.stringify(metrics));
  return metrics;
}

async function readCrawlerMetrics() {
  const store = await getCrawlerUsageStore();
  if (!store) return emptyMetrics();
  return normalizeMetrics(await readSummary(store));
}

module.exports = {
  connectBlobsLambda,
  detectCrawler,
  extractSdkDevice,
  readCrawlerMetrics,
  shouldTrackRequest,
  trackCrawlerRequest,
};
