const fs = require("fs");

const required = [
  "FIREBASE_KEY",
  "FIREBASE_AUTH_DOMAIN",
  "FIREBASE_PROJECT_ID",
  "FIREBASE_APP_ID"
];

const missing = required.filter((key) => !process.env[key]);

if (missing.length) {
  throw new Error(`Missing required Firebase env vars: ${missing.join(", ")}`);
}

const config = {
  apiKey: process.env.FIREBASE_KEY,
  authDomain: process.env.FIREBASE_AUTH_DOMAIN,
  projectId: process.env.FIREBASE_PROJECT_ID,
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET || undefined,
  messagingSenderId: process.env.FIREBASE_MESSAGING_SENDER_ID || undefined,
  appId: process.env.FIREBASE_APP_ID,
  measurementId: process.env.FIREBASE_MEASUREMENT_ID || undefined
};

const cleanConfig = Object.fromEntries(
  Object.entries(config).filter(([, value]) => Boolean(value))
);

const file = `// Generated at build time by scripts/write-firebase-config.js
// Do not commit real Firebase config manually if you are using Netlify env vars.

window.HIRING_CAFE_FIREBASE_CONFIG = ${JSON.stringify(cleanConfig, null, 2)};
`;

fs.writeFileSync("firebase-config.js", file);
console.log("Wrote firebase-config.js");
