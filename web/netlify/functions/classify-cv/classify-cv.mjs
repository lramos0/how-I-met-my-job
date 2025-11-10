// netlify/functions/classify-cv.mjs
export default async (req, context) => {
  // --- 1) Parse incoming body safely ---
  let requestBody = {};
  try {
    const raw = await req.text();                  // never throws
    requestBody = raw ? JSON.parse(raw) : {};      // tolerate empty body
  } catch (e) {
    return new Response(
      JSON.stringify({ error: "Invalid JSON in request body", detail: e.message }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // --- 2) Auth logic (unchanged, but safer) ---
  const accessKeyPub  = process.env.DBX_KEY;
  const betaPassword  = process.env.BETA_PASSWORD;

  // Your current contract uses either: 
  // - requestBody.password as the token, OR
  // - requestBody.accessKey matching betaPassword to elevate to DBX_KEY
  let accessKey = requestBody?.password || null;
  if (requestBody?.accessKey === betaPassword) {
    accessKey = accessKeyPub;
  }

  // Remove sensitive inputs we don't want to forward
  if (requestBody) {
    delete requestBody.accessKey;
    delete requestBody.pin;
    delete requestBody.password; // don't forward raw password to Databricks
  }

  if (!accessKey) {
    return new Response(
      JSON.stringify({ error: "Missing access token" }),
      { status: 401, headers: { "Content-Type": "application/json" } }
    );
  }

  // --- 3) Call Databricks; parse response as text first ---
  let upstreamRes;
  try {
    upstreamRes = await fetch(
      "https://dbc-0b26f498-9c35.cloud.databricks.com/serving-endpoints/user-score/invocations",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessKey}`
        },
        body: JSON.stringify(requestBody)
      }
    );
  } catch (e) {
    return new Response(
      JSON.stringify({ error: "Network error calling Databricks", detail: String(e) }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }

  const raw = await upstreamRes.text();            // never throws
  let parsed = null;
  try {
    parsed = raw ? JSON.parse(raw) : null;         // tolerate empty/non-JSON bodies
  } catch {
    // keep parsed = null; we’ll return raw for debugging if needed
  }

  // --- 4) Bubble up upstream errors with context ---
  if (!upstreamRes.ok) {
    return new Response(
      JSON.stringify({
        error: "Databricks request failed",
        status: upstreamRes.status,
        statusText: upstreamRes.statusText,
        body: parsed ?? raw ?? null
      }),
      { status: upstreamRes.status, headers: { "Content-Type": "application/json" } }
    );
  }

  // --- 5) Success: ensure we return valid JSON even if upstream didn't ---
  if (!parsed || typeof parsed !== "object") {
    return new Response(
      JSON.stringify({ ok: true, body: raw }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }

  return new Response(JSON.stringify(parsed), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
};
