// netlify/functions/classify-cv.mjs
export default async (req, context) => {
  try {
    // --- 1) Parse incoming body safely ---
    let requestBody = {};
    try {
      requestBody = await req.json();
    } catch (err) {
      console.error("Failed to parse incoming JSON:", err);
      return new Response(
        JSON.stringify({ error: "Invalid JSON input" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // --- 2) Auth logic ---
    const accessKeyPub = process.env.DBX_KEY;
    const betaPassword = process.env.BETA_PASSWORD;

    let accessKey = accessKeyPub;
    if (requestBody.accessKey === betaPassword) {
      accessKey = accessKeyPub;
    }

    // --- 3) Remove sensitive fields ---
    if (requestBody) {
      delete requestBody.accessKey;
      delete requestBody.pin;
      delete requestBody.password;
    }

    if (!accessKey) {
      return new Response(
        JSON.stringify({ error: "Missing access token" }),
        { status: 401, headers: { "Content-Type": "application/json" } }
      );
    }

    // --- 4) Log the full outgoing request ---
    console.log("Sending to Databricks:", JSON.stringify(requestBody, null, 2));

    // --- 5) Send to Databricks ---
    // --- Send request ---
    const response = await fetch(DATABRICKS_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessKey}`,
      },
      body: JSON.stringify(requestBody),
    });

    // --- Check status ---
    if (!response.ok) {
      const text = await response.text();
      console.error("Databricks error:", response.status, text);
      return new Response(JSON.stringify({ error: text }), { status: response.status });
    }

    // --- Parse JSON safely ---
    let result;
    try {
      result = await response.json();
    } catch (err) {
      const rawText = await response.text();
      console.error("Failed to parse JSON:", err, rawText);
      result = { error: "Invalid JSON from Databricks", raw: rawText };
    }

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
};
