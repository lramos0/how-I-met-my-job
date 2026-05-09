const { createHash, timingSafeEqual } = require("node:crypto");

const USAGE_DASHBOARD_PIN_ENV = "USAGE_DASHBOARD_PIN";
const AUTH_COOKIE_NAME = "usage_dashboard_auth";
const AUTH_COOKIE_TTL_SECONDS = 60 * 60 * 12;

function configuredPin() {
  const raw = process.env[USAGE_DASHBOARD_PIN_ENV];
  return typeof raw === "string" ? raw.trim() : "";
}

function hasConfiguredPin() {
  return configuredPin().length > 0;
}

function deriveAuthToken(pin) {
  return createHash("sha256").update(`usage-dashboard-auth::${pin}`).digest("hex");
}

function safeStringEqual(a, b) {
  const left = String(a || "");
  const right = String(b || "");
  if (left.length !== right.length) return false;
  return timingSafeEqual(Buffer.from(left), Buffer.from(right));
}

function parseCookies(cookieHeader = "") {
  const out = {};
  const raw = String(cookieHeader || "");
  if (!raw) return out;

  for (const part of raw.split(";")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const idx = trimmed.indexOf("=");
    if (idx <= 0) continue;
    const key = trimmed.slice(0, idx).trim();
    const value = trimmed.slice(idx + 1).trim();
    if (key) out[key] = decodeURIComponent(value);
  }

  return out;
}

function isHttpsRequest(request) {
  try {
    const url = new URL(request.url);
    if (url.protocol === "https:") return true;
  } catch (_) {
    // Fall through to forwarded proto.
  }

  const forwardedProto = String(request?.headers?.get("x-forwarded-proto") || "").toLowerCase();
  return forwardedProto.includes("https");
}

function secureCookieParts(request) {
  const parts = ["Path=/", "HttpOnly", "SameSite=Strict"];
  if (isHttpsRequest(request)) parts.push("Secure");
  return parts;
}

function isProvidedPinCorrect(pin) {
  const expectedPin = configuredPin();
  if (!expectedPin) return false;
  return safeStringEqual(String(pin || "").trim(), expectedPin);
}

function authStatusFromRequest(request) {
  const expectedPin = configuredPin();
  if (!expectedPin) {
    return { ok: false, reason: "missing_pin" };
  }

  const expectedToken = deriveAuthToken(expectedPin);
  const cookies = parseCookies(request?.headers?.get("cookie") || "");
  const suppliedToken = cookies[AUTH_COOKIE_NAME] || "";

  if (!suppliedToken) return { ok: false, reason: "missing_cookie" };
  if (!safeStringEqual(suppliedToken, expectedToken)) {
    return { ok: false, reason: "invalid_cookie" };
  }

  return { ok: true, reason: "authorized" };
}

function usageDashboardAuthConfigState() {
  return { configured: hasConfiguredPin() };
}

function buildAuthSetCookieHeader(request) {
  const pin = configuredPin();
  if (!pin) return null;

  const token = deriveAuthToken(pin);
  return [
    `${AUTH_COOKIE_NAME}=${encodeURIComponent(token)}`,
    `Max-Age=${AUTH_COOKIE_TTL_SECONDS}`,
    ...secureCookieParts(request),
  ].join("; ");
}

function buildAuthClearCookieHeader(request) {
  return [
    `${AUTH_COOKIE_NAME}=`,
    "Max-Age=0",
    ...secureCookieParts(request),
  ].join("; ");
}

module.exports = {
  USAGE_DASHBOARD_PIN_ENV,
  authStatusFromRequest,
  buildAuthClearCookieHeader,
  buildAuthSetCookieHeader,
  isProvidedPinCorrect,
  usageDashboardAuthConfigState,
};
