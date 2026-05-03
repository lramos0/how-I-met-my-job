# HiringCafe Vanilla Restore

A static, vanilla JavaScript restore of the jobs app using the recovered job data pool.

## What is restored

- Screenshot-style tile/card job rails by geography.
- Basic/Advanced-style header controls and filter-chip rows.
- Rich job tags from recovered columns: location, pay, employment type, seniority, industry, skills, certifications, and achievements.
- Hover overlay with Save, Mark Applied, Apply Directly, hide/report, share, View all, and See views affordances.
- Saved jobs and applied jobs state, kept private per user.
- Google sign-in/account syncing through Firebase Auth + Firestore, with a local browser fallback for development.
- Reddit-style company forums with persisted public posts, nested comments, upvotes/downvotes, and author profile pages.
- Public user profiles with display name, headline, bio, post/comment counts, vote counts, and karma-style stats.
- Static hosting anti-scraping headers for `/data/` assets where supported.

## Run locally

Because the app includes `data/jobs_restored.js`, it can open directly as a file. For best behavior, serve it:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Enable Google sign-in, saved jobs, and shared forums

1. Create or open a Firebase project.
2. Add a Web app in Firebase Project Settings.
3. Enable Authentication -> Sign-in method -> Google.
4. Create a Firestore database in **Firestore Database**, not Realtime Database. Choose Native mode if prompted, start in production mode, and pick the region you want to keep.
5. Paste your Firebase web config into `firebase-config.js`.
6. Add the Firestore indexes and starter rules in [`documentation/firestore-production.md`](documentation/firestore-production.md).

You do not need to manually create tables. Firestore collections are created by the app when the first profile, saved job, post, comment, or vote is written. The app writes private job state under `users/{uid}/private/jobState` and public community data under `forumPosts`, `forumComments`, `forumVotes`, and `users`.

## Anti-scraping note

This static restore avoids rendering all apply links as raw anchors, blocks indexing of `/data/`, and adds noindex/noarchive headers for CSV/data assets on hosts that honor `_headers` or `netlify.toml`. A determined scraper can still extract client-side data because the browser must receive it. Real protection requires a backend/API layer with server-side search, sessions, rate limits, bot checks, and signed apply URLs.
