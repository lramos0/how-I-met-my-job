(() => {
  const DATA_PATHS = ["data/jobs_restored.csv", "./data/jobs_restored.csv", "/data/jobs_restored.csv"];
  /** Canonical Job Data Pool HTTP API (see https://jobdatapool.com/) — max 500 listings per request; we merge batches up to LISTINGS_MAX. */
  const API_JOBS = "https://api.jobdatapool.com/v1/jobs";
  const LISTINGS_MAX = Math.min(1000, Number(window.JDP_LISTINGS_MAX) || 1000);
  const CACHE_KEY = "jdp_sample_jobs_v1";
  const CACHE_TTL_MS = Number(window.JDP_CACHE_TTL_MS) || 60 * 60 * 1000;
  const ACCOUNT_KEY = "hc_account_v1";
  const STATE_KEY = "hc_job_state_v1";
  const app = {
    jobs: [], filtered: [], savedOnly: false, user: null, db: null, authReady: false,
    state: { saved: {}, applied: {}, hidden: {} },
    els: {},
    filterMustHavePay: false,
    filterMustHaveEmployment: false,
    /** Lowercase substring required in listing text (from generic header chips). */
    headerKeywordBoost: null,
  };

  window.HiringCafeAuth = {
    getUser: () => app.user,
    isLoggedIn: () => !!app.user,
    promptLogin: (message) => {
      if (app.els.accountPanel) app.els.accountPanel.classList.remove("hidden");
      if (app.els.accountStatus && message) app.els.accountStatus.textContent = message;
      app.els.accountButton?.focus();
    }
  };

  document.addEventListener("DOMContentLoaded", init);

  async function init(){
    cacheEls();
    bindEvents();
    initFirebaseWhenReady();
    try {
      setSync("Loading job listings…");
      app.jobs = await loadJobsDataset();
      populateFilters(app.jobs);
      await loadState();
      applyFilters();
      if (!/Ready/i.test(app.els.syncStatus.textContent || "")) setSync("Ready");
    } catch (error) {
      showError(`Could not load the recovered jobs dataset. ${error.message || error}`);
    }
  }

  function cacheEls(){
    for (const id of ["searchInput","industryFilter","seniorityFilter","sortSelect","resultCount","syncStatus","sections","errorBox","accountButton","accountPanel","googleLogin","localLogin","logoutButton","accountStatus","savedOnly","clearFilters"]){
      app.els[id] = document.getElementById(id);
    }
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
    bindHeaderFilterChips();
    bindTopbarShortcuts();
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
      const cfg =
        window.HIRING_CAFE_FIREBASE_CONFIG ||
        window.HIRINGCAFE_FIREBASE_CONFIG ||
        {};

      const configured =
        cfg.apiKey &&
        cfg.authDomain &&
        cfg.projectId &&
        cfg.appId &&
        window.firebase;

      if (!configured) {
        restoreLocalAccount();
        app.authReady = true;
        updateAccountUi();
        return;
      }

      try {
        if (!firebase.apps.length) {
          firebase.initializeApp(cfg);
        }

        app.db = firebase.firestore();

        firebase.auth().onAuthStateChanged(async user => {
          app.user = user
            ? {
                uid: user.uid,
                name: user.displayName || user.email,
                email: user.email,
                google: true
              }
            : null;

          app.authReady = true;
          await loadState();
          updateAccountUi();
          applyFilters();
        });
      } catch (err) {
        console.warn("Firebase unavailable; falling back to local account", err);
        restoreLocalAccount();
        app.authReady = true;
        updateAccountUi();
      }
    });
  }

  function readJobsCache(){
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
    try {
      sessionStorage.setItem(CACHE_KEY, JSON.stringify({ t: Date.now(), jobs }));
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
   * Fetches at most LISTINGS_MAX listings from the API (batched at 500/request), deduped.
   */
  async function fetchSampleListingsFromApi(){
    const headers = { Accept: "application/json" };
    const perReq = 500;
    const batches = [];

    const res1 = await fetch(`${API_JOBS}?${new URLSearchParams({ limit: String(perReq), country_code: "US" })}`, { headers });
    if (!res1.ok) throw new Error(`Job Data Pool API returned ${res1.status}`);
    batches.push(...extractJobsPayload(await res1.json()));

    if (batches.length < LISTINGS_MAX) {
      const res2 = await fetch(API_JOBS, {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({ limit: perReq, country_code: "US" }),
      });
      if (res2.ok) batches.push(...extractJobsPayload(await res2.json()));
    }

    const unique = dedupeCsvRows(batches);
    return unique.slice(0, LISTINGS_MAX);
  }

  async function loadJobsDataset(){
    const params = new URLSearchParams(typeof location !== "undefined" ? location.search : "");
    const forceLocal = params.has("local") || params.has("offline") || window.JDP_USE_LOCAL_DATA === true;

    if (!forceLocal) {
      const cached = readJobsCache();
      if (cached && cached.length) {
        setSync(`Ready — ${cached.length.toLocaleString()} listings (cached sample)`);
        return cached;
      }
    }

    if (!forceLocal) {
      try {
        setSync("Fetching sample from Job Data Pool…");
        const rows = await fetchSampleListingsFromApi();
        const jobs = normalizeRows(rows);
        if (jobs.length) {
          writeJobsCache(jobs);
          setSync(`Ready — up to ${LISTINGS_MAX.toLocaleString()} sampled listings`);
          return jobs;
        }
      } catch (err) {
        console.warn("Job Data Pool API unavailable, trying bundled CSV…", err);
      }
    }

    setSync("Loading bundled dataset…");
    const csv = await loadCsv();
    const allRows = parseCsv(csv);
    const capped = allRows.slice(0, LISTINGS_MAX);
    const jobs = normalizeRows(capped);
    if (jobs.length) writeJobsCache(jobs);
    setSync(jobs.length ? `Ready — ${jobs.length.toLocaleString()} listings (local)` : "Ready");
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
    app.els.resultCount.textContent = `${app.filtered.length.toLocaleString()} jobs`;
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
        <div class="company-line"><div class="logo">${esc(initials(j.company))}</div><p class="company-copy"><b>${esc(j.company)}</b>: ${esc(firstSentence(j.summary, 120))}</p></div>
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
    const cfg =
      window.HIRING_CAFE_FIREBASE_CONFIG ||
      window.HIRINGCAFE_FIREBASE_CONFIG ||
      {};

    if (!(cfg.apiKey && window.firebase && firebase.apps.length)) {
      openDrawer("Google login not configured", "Add your Firebase config in firebase-config.js, then enable Google Authentication and Firestore. For now, use the local account to test saved listings.");
      return;
    }

    const provider = new firebase.auth.GoogleAuthProvider();
    await firebase.auth().signInWithPopup(provider);
  }
  async function localLogin(){ app.user = { uid: "local", name: "Local Account", local: true }; localStorage.setItem(ACCOUNT_KEY, JSON.stringify(app.user)); await loadState(); updateAccountUi(); applyFilters(); }
  async function logout(){ if (window.firebase && firebase.apps.length) await firebase.auth().signOut().catch(()=>{}); app.user=null; localStorage.removeItem(ACCOUNT_KEY); updateAccountUi(); }
  function restoreLocalAccount(){ try { app.user = JSON.parse(localStorage.getItem(ACCOUNT_KEY) || "null"); } catch { app.user = null; } }
  async function loadState(){
    if (app.user?.google && app.db) {
      const snap = await app.db.collection("users").doc(app.user.uid).collection("private").doc("jobState").get();
      app.state = snap.exists ? snap.data() : { saved:{}, applied:{}, hidden:{} };
    } else {
      try { app.state = JSON.parse(localStorage.getItem(STATE_KEY) || "null") || { saved:{}, applied:{}, hidden:{} }; } catch { app.state = { saved:{}, applied:{}, hidden:{} }; }
    }
  }
  async function saveState(){
    if (app.user?.google && app.db) await app.db.collection("users").doc(app.user.uid).collection("private").doc("jobState").set(app.state, { merge:true });
    else localStorage.setItem(STATE_KEY, JSON.stringify(app.state));
  }
  function updateAccountUi(){
    const logged = !!app.user;
    app.els.accountButton.textContent = logged ? (app.user.name || "Account") : "Sign up";
    app.els.accountStatus.textContent = logged ? `Signed in as ${app.user.name || app.user.email}. Saved and applied jobs are ${app.user.google ? "syncing to Firestore" : "stored locally in this browser"}.` : "Use Google login to sync saved and applied jobs. Without Firebase config, this app stores a local demo account in your browser.";
    app.els.logoutButton.classList.toggle("hidden", !logged);
    app.els.googleLogin.classList.toggle("hidden", logged);
    app.els.localLogin.classList.toggle("hidden", logged);
    window.dispatchEvent(new CustomEvent("hiringcafe:authchange", { detail: { user: app.user } }));
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
