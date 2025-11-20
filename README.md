# how-I-met-my-job — Netlify deploy notes

This repository contains a static dashboard in `webui/cvdashboard` and a Netlify Functions folder `functions/` with a mock `classify-cv` function used by the frontend.

**Netlify settings (already added)**
- `netlify.toml` at the repo root configures the publish directory and functions directory:

```
[build]
  publish = "webui/cvdashboard"
  functions = "functions"
```

Deployment options
- Web UI (no CLI required):
  1. Push the repo to GitHub.
  2. In Netlify, choose "Add new site" → "Import from Git" → select this repository.
  3. Netlify will read `netlify.toml`. Confirm the Publish directory is `webui/cvdashboard` and Functions directory is `functions`.
  4. Deploy and open the HTTPS site URL Netlify provides.

- CLI (requires Node/npm):
  1. Install Node.js (includes npm). On Windows you can use `winget install --id OpenJS.NodeJS.LTS -e` or download from nodejs.org.
  2. Install Netlify CLI: `npm install -g netlify-cli`.
  3. From repo root run (first a draft deploy):

```powershell
netlify deploy --dir="webui/cvdashboard" --functions="functions"
```

  4. For production deploy:

```powershell
netlify deploy --prod --dir="webui/cvdashboard" --functions="functions"
```

Testing the function
- The frontend calls `/.netlify/functions/classify-cv`. After deploy you can test the function with PowerShell:

```powershell
Invoke-RestMethod -Uri "https://<your-site>.netlify.app/.netlify/functions/classify-cv" \
  -Method Post \
  -Body '{"password":"craig123","inputs":[{"candidate_id":"test-1"}]}' \
  -ContentType "application/json"
```

- Or use the real curl binary (PowerShell alias `curl` maps to Invoke-WebRequest):

```powershell
curl.exe -H "Content-Type: application/json" \
  -d '{ "password":"craig123", "inputs":[{ "candidate_id":"test-1" }] }' \
  "https://<your-site>.netlify.app/.netlify/functions/classify-cv"
```

Notes
- The repo already contains a mock function at `functions/classify-cv.js` that validates the password `craig123` and returns a random `competitive_score` for each input. Replace this with your real model logic when ready.
- If you attach a custom domain like `mewannajob.com` in Netlify, call the function using that domain: `https://mewannajob.com/.netlify/functions/classify-cv` (do not mix the custom domain with the `*.netlify.app` domain in a single URL).

If you want, I can also:
- Add a small PowerShell test script `scripts/test-function.ps1` to simplify local testing.
- Replace the mock function with real evaluator logic from your training scripts.

Tell me which follow-up you want and I will add it.
