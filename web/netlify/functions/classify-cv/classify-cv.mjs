// netlify/functions/classify-cv.mjs
export default async (req, context) => {
  try {
    // 1) Parse incoming body safely
    let requestBody = {};
    try {
      requestBody = await req.json();
    } catch (err) {
      console.error("Failed to parse incoming JSON:", err);
      return new Response(JSON.stringify({ error: "Invalid JSON input" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // 2) Auth logic
    const accessKeyPub = process.env.DBX_KEY;           // Databricks PAT / SP token
    
    // always use the server-side token for Databricks calls
    const accessKey = accessKeyPub;

    // 3) Remove sensitive fields
    if (requestBody && typeof requestBody === "object") {
      delete requestBody.accessKey;
      delete requestBody.pin;
      delete requestBody.password;
    }

    if (!accessKey) {
      console.error("Missing DBX_KEY env var");
      return new Response(JSON.stringify({ error: "Missing access token" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }

    // 4) Correct Databricks Serving URL
    // Workspace host (no trailing slash)
    const WORKSPACE_HOST = "https://dbc-461c0150-500e.cloud.databricks.com";

    // IMPORTANT: replace with your actual serving endpoint name
    const ENDPOINT_NAME = process.env.DBX_ENDPOINT_NAME || "competitive_pyfunc";

    const DATABRICKS_URL = `${WORKSPACE_HOST}/api/2.0/serving-endpoints/${encodeURIComponent(
      ENDPOINT_NAME
    )}/invocations`;

    // 5) Log outgoing request (careful with PII; log shape not full content if needed)
    console.log("Calling Databricks endpoint:", ENDPOINT_NAME);
    console.log("Databricks URL:", DATABRICKS_URL);
    console.log("Outgoing payload keys:", Object.keys(requestBody || {}));

    // 6) Send to Databricks with timeout
    const controller = new AbortController();
    const timeoutMs = 25_000; // Netlify functions have limits; keep this < your function timeout
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    let response;
    try {
      response = await fetch(DATABRICKS_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessKey}`,
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });
    } catch (e) {
      clearTimeout(timeout);
      console.error("Fetch to Databricks failed:", e);
      return new Response(JSON.stringify({ error: "Failed to reach Databricks", detail: String(e) }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    } finally {
      clearTimeout(timeout);
    }

    // Read body once
    const raw = await response.text();

    if (!response.ok) {
      console.error("Databricks error:", response.status, raw.slice(0, 1000));
      return new Response(
        JSON.stringify({
          error: "Databricks invocation failed",
          status: response.status,
          body: raw.slice(0, 5000),
        }),
        { status: response.status, headers: { "Content-Type": "application/json" } }
      );
    }

    // Parse JSON safely (raw might already be JSON string)
    let result;
    try {
      result = JSON.parse(raw);
    } catch {
      result = { raw };
    }

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(JSON.stringify({ error: "Internal Server Error", detail: String(err) }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};