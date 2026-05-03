# HiringCafe UI update

Changes in this bundle:

- Decluttered the header bar: removed the small pink logo button, Basic/Advanced toggle, and location pill.
- Expanded the main search bar and updated its placeholder.
- Added `lib/fortune500.js`, which loads the 2025 Fortune 500 JSON dataset when online and exposes logo helpers and 500 company routes.
- Updated company forums to render all companies, show logos, and add a Reddit/GradCafe-style hiring metadata table per company.
- Added CSS for company logos, metadata chips, and outcomes tables.

Keep your existing `data/jobs_restored.js` or CSV in place for the jobs list.

## Login-gated posting + friendlier company routes

- Changed company forum routes from `#f` / `#f/company` to `#forums` / `#company/company-slug`.
- Kept old `#f` links as silent redirects so existing bookmarks do not break.
- Removed the visible `f/` prefix from company pages and hub cards.
- Required a signed-in account before starting a thread, commenting, or replying.
- Added a sign-in gate that opens the existing Account panel when a guest tries to post.
