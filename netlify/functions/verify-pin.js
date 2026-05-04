exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return json(405, { ok: false, error: "Method not allowed" });
  }

  const configuredPin = process.env.SITE_PIN;
  if (!configuredPin) {
    return json(500, { ok: false, error: "SITE_PIN is not configured in Netlify environment variables." });
  }

  const cookieHeader = event.headers.cookie || event.headers.Cookie || "";
  if (hasValidCookie(cookieHeader, configuredPin)) {
    return json(200, { ok: true });
  }

  let submittedPin = "";
  try {
    submittedPin = JSON.parse(event.body || "{}").pin || "";
  } catch (_) {
    return json(400, { ok: false, error: "Invalid request." });
  }

  if (safeEqual(String(submittedPin), String(configuredPin))) {
    return json(200, { ok: true }, {
      "Set-Cookie": `hc_pin=${encodeURIComponent(configuredPin)}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=86400`
    });
  }

  return json(401, { ok: false, error: "Incorrect PIN." });
};

function json(statusCode, body, headers = {}) {
  return {
    statusCode,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
      ...headers
    },
    body: JSON.stringify(body)
  };
}

function hasValidCookie(cookieHeader, configuredPin) {
  const cookies = Object.fromEntries(
    cookieHeader.split(";").map((part) => {
      const [key, ...value] = part.trim().split("=");
      return [key, decodeURIComponent(value.join("="))];
    }).filter(([key]) => key)
  );
  return cookies.hc_pin && safeEqual(cookies.hc_pin, String(configuredPin));
}

function safeEqual(a, b) {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i += 1) mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return mismatch === 0;
}
