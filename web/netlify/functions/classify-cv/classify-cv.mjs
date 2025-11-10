// netlify/functions/classify-cv.mjs
export default async (req, context) => {
  // --- 1) Parse incoming body safely ---
  const requestBody = await req.json();

  // --- 2) Auth logic (unchanged, but safer) ---
  const accessKeyPub  = process.env.DBX_KEY;
  const betaPassword  = process.env.BETA_PASSWORD;

  // Your current contract uses either: 
  // - requestBody.password as the token, OR
  // - requestBody.accessKey matching betaPassword to elevate to DBX_KEY
  let accessKey = accessKeyPub;
  if (requestBody.accessKey === betaPassword) {
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

 const response = await fetch("https://dbc-0b26f498-9c35.cloud.databricks.com/serving-endpoints/job-difficulty/invocations", {
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
