(() => {
  const DATA_PATHS = ["data/jobs_restored.csv", "./data/jobs_restored.csv", "/data/jobs_restored.csv"];
  /**
   * Local-only mode: this build intentionally does NOT call /api/jobs.
   * It loads the shipped data/jobs_restored.js or data/jobs_restored.csv bundle only.
   */
  const USE_REMOTE_JOBS = false;
  const API_JOBS = window.JDP_API_JOBS || "/api/jobs";
  const DIRECT_API_JOBS = window.JDP_DIRECT_API_JOBS || "";
  const LOCAL_LISTINGS_MAX = Math.min(4000, Number(window.JDP_LOCAL_LISTINGS_MAX) || 4000);
  const REMOTE_BATCH_SIZE = 0;
  const REMOTE_BATCHES = 0;
  const LISTINGS_MAX = Math.min(4000, Number(window.JDP_LISTINGS_MAX) || LOCAL_LISTINGS_MAX);
  const CACHE_KEY = "jdp_merged_jobs_v3";
  const CACHE_TTL_MS = Number(window.JDP_CACHE_TTL_MS) || 60 * 60 * 1000;
  const ENABLE_JOB_CACHE = window.JDP_ENABLE_JOB_CACHE === true;
  const JOB_CACHE_MAX_BYTES = Number(window.JDP_CACHE_MAX_BYTES) || 900_000;
  const ACCOUNT_KEY = "hc_account_v1";
  const LEGACY_ACCOUNT_KEY = "hc_account_v2";
  const STATE_KEY = "hc_job_state_v2";
  const PROFILE_KEY = "hc_profile_v1";
  const COUNTER_KEY = "hc_live_counters_v1";
  const app = {
    jobs: [], filtered: [], savedOnly: false, user: null, profile: null, db: null, authReady: false,
    state: { saved: {}, applied: {}, hidden: {}, posts: {}, upvotes: {}, postVotes: {} },
    counters: null,
    els: {},
    filterMustHavePay: false,
    filterMustHaveEmployment: false,
    /** Lowercase substring required in listing text (from generic header chips). */
    headerKeywordBoost: null,
  };

  window.HiringCafeAuth = {
    getUser: () => app.user,
    getProfile: () => app.profile,
    getState: () => app.state,
    getDb: () => app.db,
    getFirebase: () => (window.firebase && firebase.apps && firebase.apps.length ? firebase : null),
    getAuthReady: () => app.authReady,
    isLoggedIn: () => !!app.user,
    promptLogin: (message) => {
      if (app.els.accountPanel) app.els.accountPanel.classList.remove("hidden");
      if (app.els.accountStatus && message) app.els.accountStatus.textContent = message;
      app.els.accountButton?.focus();
    },
    saveProfile: (patch = {}) => saveProfileData(patch),
    saveForumState: async (patch = {}) => {
      app.state = { ...defaultState(), ...app.state, ...patch };
      await saveState();
      updateAccountUi();
      window.dispatchEvent(new CustomEvent("hiringcafe:forumstate", { detail: { state: app.state } }));
      return app.state;
    }
  };

  document.addEventListener("DOMContentLoaded", init);

  async function init(){
    cacheEls();
    bindEvents();
    restoreLocalAccount();
    restoreProfile();
    initFirebaseWhenReady();
    updateAccountUi();
    try {
      setSync("Loading job listings…");
      app.jobs = await loadJobsDataset();
      populateFilters(app.jobs);
      await loadState();
      restoreProfile();
      initVanityCounters();
      applyFilters();
      if (!/Ready/i.test(app.els.syncStatus.textContent || "")) setSync("Ready");
    } catch (error) {
      showError(`Could not load the recovered jobs dataset. ${error.message || error}`);
    }
  }

  function cacheEls(){
    for (const id of ["searchInput","industryFilter","seniorityFilter","sortSelect","resultCount","syncStatus","sections","errorBox","accountButton","accountPanel","googleLogin","localLogin","logoutButton","accountStatus","savedOnly","clearFilters","jobsMain","forumView","navJobs","navForums"]){
      app.els[id] = document.getElementById(id);
    }
    app.els.aiButton = document.querySelector(".ai-btn");
    app.els.menuButton = document.querySelector(".menu-btn");
    app.els.addCareerButton = document.querySelector(".green-btn");
  }

  function bindEvents(){
    app.els.searchInput.addEventListener("input", debounce(applyFilters, 120));
    app.els.industryFilter.addEventListener("change", applyFilters);
    app.els.seniorityFilter.addEventListener("change", applyFilters);
    app.els.sortSelect.addEventListener("change", applyFilters);
    app.els.clearFilters.addEventListener("click", clearAllFilters);
    app.els.savedOnly.addEventListener("click", () => { app.savedOnly = !app.savedOnly; app.els.savedOnly.classList.toggle("primary-btn", app.savedOnly); app.els.savedOnly.classList.toggle("ghost-btn", !app.savedOnly); applyFilters(); });
    app.els.accountButton.addEventListener("click", () => app.els.accountPanel.classList.toggle("hidden"));
    app.els.googleLogin.addEventListener("click", googleLogin);
    app.els.localLogin.addEventListener("click", localLogin);
    app.els.logoutButton.addEventListener("click", logout);
    app.els.aiButton?.addEventListener("click", () => { window.location.href = "https://mewannajob.com"; });
    app.els.addCareerButton?.addEventListener("click", () => openDrawer("Add Career Page", "Send this button to your company submission flow or backend endpoint. For now it is wired instead of being a dead click."));
    app.els.menuButton?.addEventListener("click", openTopMenu);
    app.els.navJobs?.addEventListener("click", (e) => { e.preventDefault(); location.hash = ""; routeMainView(); });
    app.els.navForums?.addEventListener("click", (e) => { e.preventDefault(); location.hash = "forums"; routeMainView(); });
    window.addEventListener("hashchange", routeMainView);
    bindHeaderFilterChips();
    bindTopbarShortcuts();
    routeMainView();
  }

  function scrollToolbarIntoView(){
    document.querySelector(".toolbar")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function clearHeaderChipUi(){
    document.querySelectorAll(".filters [data-filter].is-active-kw, .filters [data-filter].is-toggle-active").forEach(b => {
      b.classList.remove("is-active-kw", "is-toggle-active");
      b.removeAttribute("aria-pressed");
    });
  }

  function clearAllFilters(){
    app.els.searchInput.value = "";
    app.els.industryFilter.value = "";
    app.els.seniorityFilter.value = "";
    app.savedOnly = false;
    app.els.savedOnly.classList.remove("primary-btn");
    app.els.savedOnly.classList.add("ghost-btn");
    app.filterMustHavePay = false;
    app.filterMustHaveEmployment = false;
    app.headerKeywordBoost = null;
    clearHeaderChipUi();
    applyFilters();
  }

  function bindTopbarShortcuts(){
    document.querySelector(".brand-dot")?.addEventListener("click", () => {
      scrollToolbarIntoView();
      app.els.searchInput?.focus();
    });
    document.querySelector(".mode-pill")?.addEventListener("click", e => {
      const btn = e.target.closest("button");
      if (!btn || !e.currentTarget.contains(btn)) return;
      e.currentTarget.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      scrollToolbarIntoView();
    });
  }

  function routeMainView(){
    const raw = (location.hash || "").replace(/^#/, "").replace(/^!/, "");
    const isForum = raw === "forums" || raw === "forums/" || raw.startsWith("company/") || raw.startsWith("user/") || raw === "f" || raw.startsWith("f/");
    if (raw === "f" || raw === "f/") { location.replace("#forums"); return; }
    if (raw.startsWith("f/")) { location.replace("#company/" + encodeURIComponent(decodeURIComponent(raw.slice(2).split("/")[0] || ""))); return; }
    app.els.jobsMain?.classList.toggle("hidden", isForum);
    app.els.forumView?.classList.toggle("hidden", !isForum);
    app.els.navJobs?.classList.toggle("is-active", !isForum);
    app.els.navForums?.classList.toggle("is-active", isForum);
    document.querySelector(".filters")?.classList.toggle("hidden", isForum);
    document.querySelector(".hire-banner")?.classList.toggle("hidden", isForum);
    document.querySelector(".toolbar")?.classList.toggle("hidden", isForum);
    if (isForum) window.dispatchEvent(new CustomEvent("hiringcafe:showforums", { detail: { hash: raw } }));
  }

  function openTopMenu(){
    const signedIn = !!app.user;
    openDrawer("Menu", `
      <div class="menu-stack">
        <button type="button" data-menu-action="jobs">Jobs</button>
        <button type="button" data-menu-action="forums">Company forums</button>
        <button type="button" data-menu-action="account">${signedIn ? "Profile / account" : "Join / sign in"}</button>
        <button type="button" data-menu-action="ai">AI Search at mewannajob.com ↗</button>
        <button type="button" data-menu-action="filters">Jump to filters</button>
      </div>
    `);
    const drawer = document.querySelector(".drawer");
    drawer?.querySelectorAll("[data-menu-action]").forEach(btn => btn.addEventListener("click", () => {
      const action = btn.dataset.menuAction;
      drawer.remove();
      if (action === "jobs") { location.hash = ""; routeMainView(); }
      if (action === "forums") { location.hash = "forums"; routeMainView(); }
      if (action === "account") { app.els.accountPanel?.classList.remove("hidden"); routeMainView(); window.scrollTo({ top: 0, behavior: "smooth" }); }
      if (action === "ai") window.location.href = "https://mewannajob.com";
      if (action === "filters") { location.hash = ""; routeMainView(); scrollToolbarIntoView(); }
    }));
  }

  function bindHeaderFilterChips(){
    document.querySelectorAll(".filters [data-filter]").forEach(btn => {
      btn.addEventListener("click", () => handleHeaderFilterClick(btn));
    });
  }

  function handleHeaderFilterClick(btn){
    const label = btn.dataset.filter || "";
    scrollToolbarIntoView();

    if (label === "Industry"){
      app.headerKeywordBoost = null;
      clearKeywordOnlyChips();
      app.els.industryFilter?.focus();
      return;
    }
    if (label === "Company"){
      app.headerKeywordBoost = null;
      clearKeywordOnlyChips();
      app.els.searchInput?.focus();
      return;
    }
    if (label === "Experience"){
      app.headerKeywordBoost = null;
      clearKeywordOnlyChips();
      app.els.seniorityFilter?.focus();
      return;
    }
    if (label === "Job Titles & Keywords"){
      app.headerKeywordBoost = null;
      clearKeywordOnlyChips();
      app.els.searchInput?.focus();
      return;
    }
    if (label === "Salary"){
      app.filterMustHavePay = !app.filterMustHavePay;
      btn.classList.toggle("is-toggle-active", app.filterMustHavePay);
      btn.setAttribute("aria-pressed", app.filterMustHavePay ? "true" : "false");
      applyFilters();
      return;
    }
    if (label === "Commitment"){
      app.filterMustHaveEmployment = !app.filterMustHaveEmployment;
      btn.classList.toggle("is-toggle-active", app.filterMustHaveEmployment);
      btn.setAttribute("aria-pressed", app.filterMustHaveEmployment ? "true" : "false");
      applyFilters();
      return;
    }

    const kw = label.toLowerCase();
    if (app.headerKeywordBoost === kw){
      app.headerKeywordBoost = null;
      btn.classList.remove("is-active-kw");
    } else {
      document.querySelectorAll(".filters [data-filter].is-active-kw").forEach(b => b.classList.remove("is-active-kw"));
      app.headerKeywordBoost = kw;
      btn.classList.add("is-active-kw");
    }
    applyFilters();
  }

  function clearKeywordOnlyChips(){
    document.querySelectorAll(".filters [data-filter].is-active-kw").forEach(b => b.classList.remove("is-active-kw"));
  }

  function initFirebaseWhenReady(){
    window.addEventListener("load", async () => {
      const cfg = window.HIRINGCAFE_FIREBASE_CONFIG || {};
      const configured = cfg.apiKey && cfg.authDomain && cfg.projectId && cfg.appId && window.firebase;
      if (!configured) {
        app.authReady = true;
        restoreLocalAccount();
        updateAccountUi();
        window.dispatchEvent(new CustomEvent("hiringcafe:firebase", { detail: { db: null, user: app.user } }));
        return;
      }
      try {
        firebase.initializeApp(cfg);
        app.db = firebase.firestore();
        firebase.auth().onAuthStateChanged(async user => {
          app.user = user ? { uid: user.uid, name: user.displayName || user.email, email: user.email, google: true } : null;
          app.authReady = true;
          restoreProfile();
          if (app.user && app.db) {
            try {
              await loadCloudProfile();
              await ensureUserDocument();
            } catch (e) { console.warn("Could not load profile", e); }
          }
          await loadState();
          updateAccountUi();
          applyFilters();
          window.dispatchEvent(new CustomEvent("hiringcafe:firebase", { detail: { db: app.db, user: app.user } }));
        });
      } catch (err) {
        console.warn("Firebase unavailable; falling back to local profile", err);
        app.authReady = true;
        restoreLocalAccount();
        updateAccountUi();
        window.dispatchEvent(new CustomEvent("hiringcafe:firebase", { detail: { db: null, user: app.user } }));
      }
    });
  }

  function readJobsCache(){
    if (!ENABLE_JOB_CACHE) return null;
    try {
      const raw = sessionStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      const { t, jobs } = parsed;
      if (!t || !Array.isArray(jobs) || jobs.length === 0) return null;
      if (Date.now() - t > CACHE_TTL_MS) return null;
      return jobs;
    } catch {
      return null;
    }
  }

  function writeJobsCache(jobs){
    if (!ENABLE_JOB_CACHE || !Array.isArray(jobs) || !jobs.length) return;
    try {
      const payload = JSON.stringify({ t: Date.now(), jobs });
      if (payload.length > JOB_CACHE_MAX_BYTES) return;
      sessionStorage.setItem(CACHE_KEY, payload);
    } catch (err) {
      console.warn("Could not cache listings", err);
    }
  }

  function extractJobsPayload(payload){
    if (Array.isArray(payload)) return payload;
    if (payload && Array.isArray(payload.jobs)) return payload.jobs;
    if (payload && Array.isArray(payload.data)) return payload.data;
    return [];
  }

  function apiRowId(row){
    const id = row.id ?? row.job_id;
    if (id != null && String(id).trim()) return String(id);
    const title = row.job_title || row.title || "";
    const company = row.company_name || row.company || "";
    const loc = row.job_location || row.location || "";
    return `${title}|${company}|${loc}`;
  }

  function apiRecordToCsvRow(o){
    const get = (...keys) => {
      for (const k of keys) {
        if (o[k] != null && o[k] !== "") return o[k];
      }
      return "";
    };
    return {
      id: get("id"),
      job_title: get("job_title", "title"),
      company_name: get("company_name", "company"),
      job_location: get("job_location", "location"),
      job_seniority_level: get("job_seniority_level", "seniority"),
      job_employment_type: get("job_employment_type", "employment_type"),
      job_industries: get("job_industries", "industries"),
      industries: get("industries", "job_industries"),
      job_summary: get("job_summary", "summary"),
      job_base_pay_range: get("job_base_pay_range", "pay_range"),
      job_posted_date: get("job_posted_date", "posted_date"),
      competitiveness_score: get("competitiveness_score", "score"),
      skills: get("skills"),
      certifications: get("certifications"),
      achievements: get("achievements"),
      url: get("url"),
      apply_link: get("apply_link", "apply_url"),
      country_code: get("country_code"),
      ingest_utc_date: get("ingest_utc_date"),
    };
  }

  function dedupeCsvRows(rows){
    const seen = new Map();
    for (const r of rows) {
      const row = typeof r === "object" && r ? apiRecordToCsvRow(r) : {};
      const key = apiRowId(row);
      if (!seen.has(key)) seen.set(key, row);
    }
    return [...seen.values()];
  }

  /**
   * Fetches exactly the requested four 500-listing pages from a same-origin proxy.
   * Browsers cannot read Job Data Pool directly unless their API sends CORS headers.
   */
  async function fetchSampleListingsFromApi(){
    const headers = { Accept: "application/json" };
    const batches = [];
    const failures = [];
    const endpoints = [API_JOBS];
    if (DIRECT_API_JOBS) endpoints.push(DIRECT_API_JOBS);

    for (let batch = 0; batch < REMOTE_BATCHES; batch++) {
      const offset = batch * REMOTE_BATCH_SIZE;
      const params = new URLSearchParams({
        limit: String(REMOTE_BATCH_SIZE),
        offset: String(offset),
        page: String(batch + 1),
        country_code: "US"
      });

      let loadedThisBatch = false;
      for (const endpoint of endpoints) {
        try {
          const res = await fetch(`${endpoint}?${params}`, { headers, cache: "no-store", credentials: "same-origin" });
          if (!res.ok) throw new Error(`GET ${endpoint} batch ${batch + 1} returned ${res.status}`);
          const rows = extractJobsPayload(await res.json());
          if (!rows.length) throw new Error(`GET ${endpoint} batch ${batch + 1} returned no jobs`);
          batches.push(...rows);
          loadedThisBatch = true;
          break;
        } catch (err) {
          failures.push(err.message || String(err));
        }
      }

      if (!loadedThisBatch) break;
    }

    if (!batches.length && failures.length) throw new Error(failures[0]);
    return dedupeCsvRows(batches).slice(0, REMOTE_BATCH_SIZE * REMOTE_BATCHES);
  }

  async function loadLocalJobsRows(){
    const csv = await loadCsv();
    return parseCsv(csv).slice(0, LOCAL_LISTINGS_MAX);
  }

  async function loadJobsDataset(){
    // Local-only build: avoid /api/jobs entirely so rate limits or 502s cannot affect load.
    // data/jobs_restored.js is loaded before app.js in index.html and exposes RESTORED_JOBS_CSV.
    setSync(`Loading ${LOCAL_LISTINGS_MAX.toLocaleString()} bundled listings…`);
    const localRows = await loadLocalJobsRows();
    const jobs = normalizeRows(dedupeCsvRows(localRows).slice(0, LISTINGS_MAX));
    if (jobs.length) writeJobsCache(jobs);
    setSync(jobs.length ? `Ready — ${jobs.length.toLocaleString()} local listings` : "Ready — no bundled listings found");
    return jobs;
  }

  async function loadCsv(){
    if (typeof window.RESTORED_JOBS_CSV === "string" && window.RESTORED_JOBS_CSV.trim()) return window.RESTORED_JOBS_CSV;
    let lastError = null;
    for (const path of DATA_PATHS){
      try {
        const res = await fetch(path, { cache: "no-store" });
        if (!res.ok) throw new Error(`${path} returned ${res.status}`);
        return await res.text();
      } catch (err) { lastError = err; }
    }
    throw lastError || new Error("No CSV source found.");
  }

  function parseCsv(text){
    const rows = []; let row = [], cell = "", q = false;
    for (let i=0;i<text.length;i++){
      const c=text[i], n=text[i+1];
      if (c === '"') { if (q && n === '"') { cell += '"'; i++; } else q = !q; }
      else if (c === ',' && !q) { row.push(cell); cell=""; }
      else if ((c === '\n' || c === '\r') && !q) { if (c === '\r' && n === '\n') i++; row.push(cell); if (row.some(x => x.trim() !== "")) rows.push(row); row=[]; cell=""; }
      else cell += c;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    const headers = rows.shift().map(h => h.trim());
    return rows.map(r => Object.fromEntries(headers.map((h,i) => [h, (r[i] || "").trim()])));
  }

  function normalizeRows(rows){
    return rows.filter(r => r.job_title && r.company_name).map((r, idx) => {
      const pay = r.job_base_pay_range || "";
      return {
        id: r.id || `job-${idx}`,
        title: clean(r.job_title), company: clean(r.company_name), location: cleanLocation(r.job_location),
        seniority: clean(r.job_seniority_level), employment: clean(r.job_employment_type),
        industries: splitTags(r.industries || r.job_industries), summary: clean(r.job_summary),
        pay: clean(pay), posted: r.job_posted_date || r.ingest_utc_date || "", score: Number(r.competitiveness_score || 0),
        skills: splitTags(r.skills), certifications: splitTags(r.certifications), achievements: splitTags(r.achievements),
        url: r.url || "", apply: r.apply_link || r.url || "", country: r.country_code || "US",
        companyDomain: companyDomain(clean(r.company_name), r.apply_link || r.url || ""),
        cityKey: cityKey(cleanLocation(r.job_location))
      };
    });
  }

  function clean(v){ return String(v || "").replace(/\s+/g," ").replace(/^nan$/i,"").trim(); }
  function cleanLocation(v){ return clean(v).replace(/^Location:\s*/i, "").replace(/,\s*US,?\s*\d*$/i, ", United States").replace(/,\s*US$/i, ", United States"); }
  function splitTags(v){ return clean(v).split(/;|,\s(?=[A-Z0-9])/).map(x => x.trim()).filter(Boolean).slice(0, 10); }
  function cityKey(loc){
    const s = loc.toLowerCase();
    if (/los angeles|lynwood|inglewood|lax|santa monica|burbank|pasadena/.test(s)) return "Los Angeles";
    if (/new york|brooklyn|queens|manhattan/.test(s)) return "New York";
    if (/san francisco|oakland|san jose|palo alto|redwood|mountain view/.test(s)) return "Bay Area";
    if (/chicago/.test(s)) return "Chicago";
    if (/austin/.test(s)) return "Austin";
    return "United States";
  }

  function populateFilters(jobs){
    fillSelect(app.els.industryFilter, topValues(jobs.flatMap(j => j.industries), 35));
    fillSelect(app.els.seniorityFilter, topValues(jobs.map(j => j.seniority).filter(Boolean), 20));
  }
  function topValues(values, limit){
    const counts = new Map(); values.forEach(v => counts.set(v, (counts.get(v)||0)+1));
    return [...counts.entries()].sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0])).slice(0, limit).map(([v])=>v);
  }
  function fillSelect(sel, values){ values.forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=v; sel.append(o); }); }

  function jobHaystack(j){
    return [j.title,j.company,j.location,j.summary,j.pay,j.seniority,j.employment,...j.industries,...j.skills,...j.certifications,...j.achievements].join(" ").toLowerCase();
  }

  function applyFilters(){
    const q = app.els.searchInput.value.trim().toLowerCase();
    const industry = app.els.industryFilter.value;
    const seniority = app.els.seniorityFilter.value;
    const boost = app.headerKeywordBoost;
    app.filtered = app.jobs.filter(j => {
      if (app.state.hidden[j.id]) return false;
      if (app.savedOnly && !app.state.saved[j.id]) return false;
      if (industry && !j.industries.includes(industry)) return false;
      if (seniority && j.seniority !== seniority) return false;
      if (app.filterMustHavePay && !String(j.pay || "").trim()) return false;
      if (app.filterMustHaveEmployment && !String(j.employment || "").trim()) return false;
      const hay = jobHaystack(j);
      if (boost && !hay.includes(boost)) return false;
      if (!q) return true;
      return hay.includes(q);
    });
    sortJobs(app.filtered);
    render();
  }

  function sortJobs(jobs){
    const mode = app.els.sortSelect.value;
    if (mode === "score") jobs.sort((a,b)=>b.score-a.score || freshness(b)-freshness(a));
    else if (mode === "pay") jobs.sort((a,b)=>payMax(b.pay)-payMax(a.pay));
    else if (mode === "company") jobs.sort((a,b)=>a.company.localeCompare(b.company));
    else jobs.sort((a,b)=>freshness(b)-freshness(a));
  }
  function freshness(j){ return Date.parse(j.posted || "") || 0; }
  function payMax(pay){ const nums=(pay.match(/\$?([0-9]+(?:\.[0-9]+)?)(k)?/ig)||[]).map(x=>{ const k=/k/i.test(x); const n=Number(x.replace(/[^0-9.]/g,"")); return k?n*1000:n; }); return nums.length?Math.max(...nums):0; }

  function render(){
    app.els.resultCount.textContent = homepageCountLabel(app.filtered.length);
    const groups = buildGroups(app.filtered);
    app.els.sections.innerHTML = groups.map(([name, jobs], i) => sectionHtml(name, jobs, i)).join("");
    bindRenderedActions();
  }
  function buildGroups(jobs){
    if (app.savedOnly) return [["Saved Job Listings", jobs.slice(0, 80)]];
    const used = new Set();
    const groupNames = ["United States","Los Angeles","New York","Bay Area","Chicago","Austin"];
    const groups = [];
    for (const name of groupNames){
      const list = jobs.filter(j => (name === "United States" ? true : j.cityKey === name) && !used.has(j.id)).slice(0, 18);
      list.forEach(j => used.add(j.id));
      if (list.length) groups.push([`Latest Jobs in ${name}`, list]);
    }
    return groups.length ? groups : [["Latest Jobs", jobs.slice(0, 24)]];
  }
  function sectionHtml(title, jobs, idx){
    return `<section class="job-section"><div class="section-head"><h2>${esc(title)} <a href="#" aria-hidden="true">→</a></h2><div class="section-arrows"><button class="arrow" data-scroll="left" data-rail="rail-${idx}">‹</button><button class="arrow" data-scroll="right" data-rail="rail-${idx}">›</button></div></div><div id="rail-${idx}" class="rail">${jobs.map(cardHtml).join("")}</div></section>`;
  }
  function cardHtml(j){
    const saved = !!app.state.saved[j.id], applied = !!app.state.applied[j.id];
    const tags = [locTag(j.location), payTag(j.pay), strongTag(j.employment || "Full Time"), strongTag(j.seniority), expTag(j.achievements[0]), ...j.certifications.slice(0,1).map(tag), ...j.industries.slice(0,1).map(tag)].filter(Boolean).join("");
    const skillText = [...j.skills.slice(0,4), ...j.certifications.slice(0,2)].join(", ");
    return `<article class="job-card ${saved?'saved':''} ${applied?'applied':''}" data-id="${escAttr(j.id)}">
      <div class="hover-actions">
        <div class="hover-top"><button class="round" data-action="share" title="Share">⇧</button><button class="save" data-action="save">${saved?'Saved':'Save'}</button><button class="mark" data-action="applied">${applied?'Applied':'Mark Applied'}</button></div>
        <div></div>
        <div class="hover-bottom"><button class="apply" data-action="apply">Apply Directly</button><div class="icon-actions"><button data-action="hide" title="Hide">⊘</button><button data-action="report" title="Report">⚑</button></div></div>
      </div>
      <div class="card-main">
        <div class="title-row"><h3 class="job-title">${esc(j.title)}</h3><span class="time">◷ ${timeAgo(j.posted)}</span></div>
        <div class="tag-row">${tags}</div>
        <div class="company-line"><div class="logo logo-img-wrap">${companyLogoHtml(j, 40)}</div><p class="company-copy"><b>${esc(j.company)}</b>: ${esc(firstSentence(j.summary, 120))}</p></div>
        <p class="summary"><span class="icon">▣</span>${esc(snippet(j.summary, 260))}</p>
        ${skillText ? `<p class="skills-line"><span class="icon">🛠</span>${esc(skillText)}</p>` : ""}
      </div>
      <div class="card-footer"><div class="footer-row"><button data-action="apply">Job Posting ↗</button><button data-action="details">View all</button></div><div class="views">⌁ See views</div></div>
    </article>`;
  }
  const tag = v => v ? `<span class="tag">${esc(v)}</span>` : "";
  const locTag = v => v ? `<span class="tag">⌾ ${esc(v)}</span>` : "";
  const payTag = v => v ? `<span class="tag pay">${esc(shortPay(v))}</span>` : "";
  const strongTag = v => v ? `<span class="tag strong">${esc(v)}</span>` : "";
  const expTag = v => v ? `<span class="tag exp">${esc(v.length > 14 ? v.slice(0,14)+"…" : v)}</span>` : "";

  function bindRenderedActions(){
    document.querySelectorAll("[data-scroll]").forEach(btn => btn.addEventListener("click", () => {
      const rail = document.getElementById(btn.dataset.rail); if (rail) rail.scrollBy({ left: btn.dataset.scroll === "right" ? 620 : -620, behavior: "smooth" });
    }));
    document.querySelectorAll(".job-card [data-action]").forEach(btn => btn.addEventListener("click", async (e) => {
      e.preventDefault(); e.stopPropagation();
      const card = btn.closest(".job-card"); const id = card.dataset.id; const job = app.jobs.find(j => j.id === id); const action = btn.dataset.action;
      if (action === "save") { app.state.saved[id] ? delete app.state.saved[id] : app.state.saved[id] = Date.now(); await saveState(); applyFilters(); }
      if (action === "applied") { app.state.applied[id] ? delete app.state.applied[id] : app.state.applied[id] = Date.now(); await saveState(); applyFilters(); }
      if (action === "hide") { app.state.hidden[id] = Date.now(); await saveState(); applyFilters(); }
      if (action === "share") shareJob(job);
      if (action === "report") openDrawer("Report job", `Flagged ${job.title} at ${job.company}. Wire this button to your moderation endpoint when backend services are restored.`);
      if (action === "details") openDetails(job);
      if (action === "apply") hydrateAndOpen(job);
    }));
  }

  function hydrateAndOpen(job){
    const href = job.apply || job.url;
    if (!href) return openDrawer("No application link", "This recovered record did not include an apply URL.");
    window.open(href, "_blank", "noopener,noreferrer");
  }
  function openDetails(job){
    openDrawer(job.title, `<b>${esc(job.company)}</b><br>${esc(job.location)}<br><br>${esc(job.summary.slice(0,1200))}<br><br><b>Tags:</b> ${esc([...job.industries,...job.skills,...job.certifications].slice(0,18).join(", "))}`);
  }
  function openDrawer(title, html){
    document.querySelector(".drawer")?.remove();
    const d=document.createElement("aside"); d.className="drawer"; d.innerHTML=`<button class="close" aria-label="Close">×</button><h3>${esc(title)}</h3><p>${html}</p>`;
    d.querySelector(".close").addEventListener("click",()=>d.remove()); document.body.append(d);
  }
  function shareJob(job){
    const text = `${job.title} at ${job.company}`;
    if (navigator.share) navigator.share({ title: text, text }).catch(()=>{});
    else navigator.clipboard?.writeText(text).then(()=>openDrawer("Copied", "Job summary copied to clipboard."));
  }

  async function googleLogin(){
    const cfg = window.HIRINGCAFE_FIREBASE_CONFIG || {};
    if (!(cfg.apiKey && window.firebase && firebase.apps.length)) {
      openDrawer("Google sign-in not configured", "Add your Firebase config in firebase-config.js, then enable Google Authentication and Firestore. For now, try a local profile to test saved listings.");
      return;
    }
    const provider = new firebase.auth.GoogleAuthProvider();
    await firebase.auth().signInWithPopup(provider);
  }
  async function localLogin(){
    const existing = readJson(ACCOUNT_KEY, null) || readJson(LEGACY_ACCOUNT_KEY, null);
    app.user = existing || { uid: `local-${Date.now().toString(36)}`, name: "Local Profile", local: true, createdAt: Date.now() };
    app.user.local = true;
    if (!app.user.uid) app.user.uid = `local-${Date.now().toString(36)}`;
    localStorage.setItem(ACCOUNT_KEY, JSON.stringify(app.user));
    localStorage.removeItem(LEGACY_ACCOUNT_KEY);
    restoreProfile();
    await loadState();
    updateAccountUi();
    applyFilters();
    app.els.accountPanel?.classList.remove("hidden");
    window.dispatchEvent(new CustomEvent("hiringcafe:authchange", { detail: { user: app.user, profile: app.profile, state: app.state } }));
  }
  async function logout(){
    if (window.firebase && firebase.apps.length) await firebase.auth().signOut().catch(()=>{});
    app.user = null;
    app.profile = null;
    localStorage.removeItem(ACCOUNT_KEY);
    localStorage.removeItem(LEGACY_ACCOUNT_KEY);
    updateAccountUi();
    window.dispatchEvent(new CustomEvent("hiringcafe:authchange", { detail: { user: null, profile: null, state: app.state } }));
  }
  function restoreLocalAccount(){
    app.user = readJson(ACCOUNT_KEY, null) || readJson(LEGACY_ACCOUNT_KEY, null);
    if (app.user) {
      if (!app.user.uid) app.user.uid = `local-${Date.now().toString(36)}`;
      app.user.local = app.user.local !== false && !app.user.google;
      localStorage.setItem(ACCOUNT_KEY, JSON.stringify(app.user));
      localStorage.removeItem(LEGACY_ACCOUNT_KEY);
    }
  }
  async function loadState(){
    if (app.user?.google && app.db) {
      const snap = await app.db.collection("users").doc(app.user.uid).collection("private").doc("jobState").get();
      app.state = { ...defaultState(), ...(snap.exists ? snap.data() : {}) };
    } else {
      try { app.state = { ...defaultState(), ...(JSON.parse(localStorage.getItem(STATE_KEY) || "null") || {}) }; } catch { app.state = defaultState(); }
    }
  }
  async function saveState(){
    if (app.user?.google && app.db) await app.db.collection("users").doc(app.user.uid).collection("private").doc("jobState").set(app.state, { merge:true });
    else localStorage.setItem(STATE_KEY, JSON.stringify(app.state));
  }
  function updateAccountUi(){
    const logged = !!app.user;
    if (app.els.accountButton) {
      app.els.accountButton.textContent = logged ? (app.profile?.displayName || app.user.name || "Profile") : "Join";
      app.els.accountButton.setAttribute("aria-label", logged ? "Open your profile and account panel" : "Join or sign in");
    }
    if (app.els.accountStatus) app.els.accountStatus.textContent = logged
      ? `Signed in as ${app.profile?.displayName || app.user.name || app.user.email}. Saved jobs stay private; ${app.user.google ? "forum posts, comments, votes, and profile data sync through Firestore" : "local forum activity stays in this browser"}.`
      : "Create a profile to save jobs, post in company forums, comment, and vote. Google syncs across devices; local mode is just for this browser.";
    app.els.logoutButton?.classList.toggle("hidden", !logged);
    app.els.googleLogin?.classList.toggle("hidden", logged);
    app.els.localLogin?.classList.toggle("hidden", logged);
    renderProfilePanel();
    window.dispatchEvent(new CustomEvent("hiringcafe:authchange", { detail: { user: app.user, profile: app.profile, state: app.state } }));
  }

  function renderProfilePanel(){
    const panel = app.els.accountPanel;
    if (!panel) return;
    let card = panel.querySelector("#profileCard");
    if (!card) {
      card = document.createElement("div");
      card.id = "profileCard";
      card.className = "profile-card";
      panel.append(card);
    }
    if (!app.user) {
      card.innerHTML = `<h3>Your profile</h3><p>Join or try a local profile to unlock saved jobs, posts, comments, and voting.</p>`;
      return;
    }
    const profile = app.profile || defaultProfile();
    const localPosts = Object.values(app.state.posts || {}).filter(p => p.authorUid === app.user.uid).length;
    const localComments = Object.values(app.state.comments || {}).filter(c => c.authorUid === app.user.uid).length;
    const localVotes = Object.keys(app.state.upvotes || {}).filter(k => k.startsWith(`${app.user.uid}:`)).length;
    const stats = { ...defaultProfileStats(), ...(profile.stats || {}) };
    const posts = Math.max(Number(stats.posts || 0), localPosts);
    const comments = Math.max(Number(stats.comments || 0), localComments);
    const votes = Math.max(Number(stats.votesCast || 0), localVotes);
    card.innerHTML = `
      <h3>Your profile</h3>
      <label>Display name <input id="profileDisplayName" value="${escAttr(profile.displayName || '')}" placeholder="Display name"></label>
      <label>Headline <input id="profileHeadline" value="${escAttr(profile.headline || '')}" placeholder="Software engineer, student, recruiter..."></label>
      <label>Bio <textarea id="profileBio" placeholder="What should people know when they see your posts?">${esc(profile.bio || "")}</textarea></label>
      <div class="profile-stats"><span>${posts} posts</span><span>${comments} comments</span><span>${votes} votes cast</span><span>${Object.keys(app.state.saved || {}).length} saved jobs</span></div>
      <button class="primary-btn" id="saveProfileBtn" type="button">Save profile</button>
    `;
    card.querySelector("#saveProfileBtn")?.addEventListener("click", saveProfileFromPanel);
  }

  function defaultProfile(){
    return {
      uid: app.user?.uid || "guest",
      displayName: app.user?.name || app.user?.email || "Local Profile",
      headline: "",
      bio: "",
      stats: defaultProfileStats(),
      createdAt: app.user?.createdAt || Date.now(),
      updatedAt: Date.now(),
    };
  }

  function defaultProfileStats(){
    return { posts: 0, comments: 0, votesCast: 0, karma: 0 };
  }

  function profileStorageKey(){
    return app.user?.uid ? `${PROFILE_KEY}:${app.user.uid}` : PROFILE_KEY;
  }

  function restoreProfile(){
    if (!app.user) { app.profile = null; return; }
    const saved = readJson(profileStorageKey(), null) || readJson(PROFILE_KEY, null);
    app.profile = sanitizeProfile(saved || {});
    localStorage.setItem(profileStorageKey(), JSON.stringify(app.profile));
  }

  async function loadCloudProfile(){
    if (!(app.user?.google && app.db)) return;
    const profileSnap = await app.db.collection("users").doc(app.user.uid).get();
    if (!profileSnap.exists) return;
    const data = profileSnap.data() || {};
    app.profile = sanitizeProfile({
      ...(data.profile || {}),
      stats: { ...defaultProfileStats(), ...(data.profile?.stats || {}), ...(data.stats || {}) }
    });
    localStorage.setItem(profileStorageKey(), JSON.stringify(app.profile));
  }

  async function ensureUserDocument(){
    if (!(app.user?.google && app.db)) return;
    const profile = sanitizeProfile(app.profile || {});
    app.profile = profile;
    await app.db.collection("users").doc(app.user.uid).set({
      uid: app.user.uid,
      profile,
      stats: profile.stats || defaultProfileStats(),
      updatedAt: Date.now()
    }, { merge: true });
  }

  function sanitizeProfile(profile){
    const base = { ...defaultProfile(), ...(profile || {}) };
    const displayName = String(base.displayName || app.user?.name || app.user?.email || "Community Member").trim().slice(0, 80);
    const headline = String(base.headline || "").trim().slice(0, 120);
    const bio = String(base.bio || "").trim().slice(0, 500);
    return {
      ...base,
      uid: app.user?.uid || base.uid || "guest",
      displayName: displayName || "Community Member",
      headline,
      bio,
      stats: { ...defaultProfileStats(), ...(base.stats || {}) },
      updatedAt: base.updatedAt || Date.now()
    };
  }

  async function saveProfileData(patch = {}){
    if (!app.user) return null;
    app.profile = sanitizeProfile({ ...app.profile, ...patch, updatedAt: Date.now() });
    app.user.name = app.profile.displayName;
    localStorage.setItem(ACCOUNT_KEY, JSON.stringify(app.user));
    localStorage.setItem(profileStorageKey(), JSON.stringify(app.profile));
    localStorage.setItem(PROFILE_KEY, JSON.stringify(app.profile));
    if (app.user.google && app.db) {
      const { stats, ...profileForCloud } = app.profile;
      await app.db.collection("users").doc(app.user.uid).set({
        uid: app.user.uid,
        profile: profileForCloud,
        updatedAt: Date.now()
      }, { merge: true });
    }
    return app.profile;
  }

  async function saveProfileFromPanel(){
    if (!app.user) return;
    await saveProfileData({
      displayName: document.getElementById("profileDisplayName")?.value.trim() || "Local Profile",
      headline: document.getElementById("profileHeadline")?.value.trim() || "",
      bio: document.getElementById("profileBio")?.value.trim() || "",
    });
    updateAccountUi();
    openDrawer("Profile saved", "Your profile is ready for forum posts and upvote tracking.");
  }

  function defaultState(){ return { saved: {}, applied: {}, hidden: {}, posts: {}, comments: {}, upvotes: {}, postVotes: {} }; }
  function readJson(key, fallback){ try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch { return fallback; } }

  function initVanityCounters(){
    app.counters = readJson(COUNTER_KEY, null) || { companies: 5600000, jobs: 3100000, updatedAt: Date.now() };
    tickVanityCounters();
    setInterval(tickVanityCounters, 45000);
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("service-worker.js").then(reg => {
        reg.active?.postMessage({ type: "HC_COUNTERS", counters: app.counters });
      }).catch(err => console.warn("Service worker registration skipped", err));
    }
  }

  function tickVanityCounters(){
    const now = Date.now();
    const last = app.counters?.updatedAt || now;
    const minutes = Math.max(1, Math.floor((now - last) / 60000));
    app.counters = app.counters || { companies: 5600000, jobs: 3100000, updatedAt: now };
    app.counters.jobs += Math.min(2500, minutes * 17 + Math.floor(Math.random() * 19));
    app.counters.companies += Math.min(900, minutes * 3 + Math.floor(Math.random() * 7));
    app.counters.updatedAt = now;
    localStorage.setItem(COUNTER_KEY, JSON.stringify(app.counters));
    if (app.filtered) app.els.resultCount.textContent = homepageCountLabel(app.filtered.length);
    navigator.serviceWorker?.controller?.postMessage({ type: "HC_COUNTERS", counters: app.counters });
  }

  function homepageCountLabel(realFiltered){
    const counters = app.counters || { companies: 5600000, jobs: 3100000 };
    return `${formatLarge(counters.jobs)} jobs · ${formatLarge(counters.companies)} companies · ${Number(realFiltered || 0).toLocaleString()} loaded`;
  }

  function formatLarge(n){ return Math.round(Number(n || 0)).toLocaleString(); }

  function companyDomain(company, url){
    const fromUrl = domainFromUrl(url);
    if (fromUrl) return fromUrl;
    const registry = window.Fortune500?.list?.().find(c => c.name && clean(c.name).toLowerCase() === clean(company).toLowerCase());
    if (registry?.domain) return registry.domain;
    const overrides = window.HC_DOMAIN_OVERRIDES || {};
    if (overrides[company]) return overrides[company];
    return clean(company).toLowerCase().replace(/&/g,"and").replace(/\b(company|companies|corporation|corp|inc|llc|holdings|group|international|systems|technologies|technology|services|the)\b/g,"").replace(/[^a-z0-9]/g,"") + ".com";
  }

  function domainFromUrl(url){
    try {
      const host = new URL(url).hostname.replace(/^www\./, "");
      return host || "";
    } catch { return ""; }
  }

  function companyLogoHtml(job, size){
    const domain = job.companyDomain || companyDomain(job.company, job.apply || job.url);
    const fallback = fallbackLogoSvg(job.company, size);
    const src = `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=${size * 2}`;
    return `<img class="company-logo-img" src="${escAttr(src)}" width="${size}" height="${size}" alt="${escAttr(job.company)} logo" loading="lazy" decoding="async" onerror="this.onerror=null;this.src='${fallback.replace(/'/g, "%27")}';" />`;
  }

  function fallbackLogoSvg(company, size){
    const label = initials(company);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><rect width="100%" height="100%" rx="12" fill="#fff7ed"/><text x="50%" y="54%" text-anchor="middle" dominant-baseline="middle" font-family="Arial, sans-serif" font-size="${Math.max(12, size/3)}" font-weight="800" fill="#7c2d12">${esc(label)}</text></svg>`;
    return "data:image/svg+xml;charset=UTF-8," + encodeURIComponent(svg);
  }

  function setSync(s){ app.els.syncStatus.textContent = s; }
  function showError(msg){ app.els.errorBox.textContent = msg; app.els.errorBox.classList.remove("hidden"); setSync("Dataset error"); }
  function esc(s){ return String(s||"").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function escAttr(s){ return esc(s).replace(/`/g,"&#96;"); }
  function initials(s){ return (s||"?").split(/\s+/).filter(Boolean).slice(0,2).map(x=>x[0]).join("").toUpperCase(); }
  function firstSentence(s, max){ const t=clean(s); const m=t.match(/^.{20,}?[.!?](\s|$)/); return snippet(m?m[0]:t,max); }
  function snippet(s, max){ s=clean(s); return s.length>max ? s.slice(0,max-1).trim()+"…" : s; }
  function shortPay(s){ return clean(s).replace(/per hour/ig,"/hr").replace(/hourly/ig,"/hr").replace(/\s+/g," ").slice(0,28); }
  function timeAgo(date){ const t=Date.parse(date||""); if(!t) return "5h"; const h=Math.max(1, Math.round((Date.now()-t)/36e5)); if(h<24) return `${h}h`; const d=Math.round(h/24); return d<30?`${d}d`:`${Math.round(d/30)}mo`; }
  function debounce(fn, ms){ let id; return (...args)=>{ clearTimeout(id); id=setTimeout(()=>fn(...args), ms); }; }
})();
