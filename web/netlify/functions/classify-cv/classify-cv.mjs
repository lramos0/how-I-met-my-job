// netlify/functions/classify-cv.mjs
export default async (req, context) => {
  // --- 1) Parse incoming body safely ---
  let requestBody = {};
  try {
    requestBody = await req.json();
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Invalid JSON in request body" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // --- 2) Auth logic ---
  const accessKeyPub = process.env.DBX_KEY;
  const betaPassword = process.env.BETA_PASSWORD;

  // Determine which key to use
  let accessKey = accessKeyPub;
  if (requestBody.accessKey && requestBody.accessKey === betaPassword) {
    accessKey = accessKeyPub;
  }

  if (!accessKey) {
    return new Response(
      JSON.stringify({ error: "Missing access token" }),
      { status: 401, headers: { "Content-Type": "application/json" } }
    );
  }

  // --- 3) Redact sensitive fields before logging ---
  const safeBody = { ...requestBody };
  delete safeBody.accessKey;
  delete safeBody.pin;
  // Keep password if Databricks requires it; remove only from logs
  const logBody = { ...safeBody };
  delete logBody.password;
  console.log("Sending to Databricks:", logBody);

  // --- 4) Send request to Databricks ---
  let dbResult = {};
  try {
    const response = await fetch(
      "https://dbc-0b26f498-9c35.cloud.databricks.com/serving-endpoints/job-difficulty/invocations",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessKey}`,
        },
        body: JSON.stringify(safeBody),
      }
    );

    const text = await response.text();
    if (!text) {
      dbResult = { warning: "Databricks returned empty response" };
    } else {
      try {
        dbResult = JSON.parse(text);
      } catch {
        dbResult = { error: "Databricks returned invalid JSON", raw: text };
      }
    }

    if (!response.ok) {
      return new Response(JSON.stringify(dbResult), {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    }
  } catch (err) {
    console.error("Error contacting Databricks:", err);
    return new Response(
      JSON.stringify({ error: "Failed to contact Databricks" }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }

  // --- 5) Return safe result ---
  return new Response(JSON.stringify(dbResult), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};

