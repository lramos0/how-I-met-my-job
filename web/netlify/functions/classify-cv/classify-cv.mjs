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
    const response = await fetch(
      "https://dbc-0b26f498-9c35.cloud.databricks.com/serving-endpoints/job-difficulty/invocations",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessKey}`,
        },
        body: JSON.stringify(requestBody),
      }
    );

    // --- 6) Read raw response text ---
    const rawText = await response.text();
    console.log("Databricks raw response:", rawText);

    let result;
    try {
      result = rawText ? JSON.parse(rawText) : { warning: "Databricks returned empty response" };
    } catch (err) {
      console.error("Failed to parse Databricks JSON:", err);
      result = { error: "Invalid JSON from Databricks", raw: rawText };
    }

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(
      JSON.stringify({ error: "Internal server error" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
};
