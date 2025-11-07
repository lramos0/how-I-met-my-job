export default async (req, context) => {
  const requestBody = await req.json();
  const accessKeyPub = process.env.DBX_KEY;
  const betaPassword = process.env.BETA_PASSWORD;
  let accessKey = requestBody.password;
  if (requestBody.accessKey == betaPassword) {
    accessKey = accessKeyPub;
  }
  delete requestBody.accessKey;
  delete requestBody.pin;

  if (!accessKey) {
    return new Response(JSON.stringify({ error: "Missing access token" }), {
      status: 401,
      headers: { "Content-Type": "application/json" }
    });
  }

  const response = await fetch("https://dbc-0b26f498-9c35.cloud.databricks.com/serving-endpoints/user-score/invocations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessKey}`
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    return new Response(JSON.stringify({ error: "Databricks request failed" }), {
      status: response.status,
      headers: { "Content-Type": "application/json" }
    });
  }

  const result = await response.json();

  return new Response(JSON.stringify(result), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
};