/* jobs.js
 *
 * Responsibilities:
 *  - Load resume (localStorage or ./parsed_resume.json fallback)
 *  - Fetch jobs from Netlify function /.netlify/functions/get-jobs
 *  - Normalize + store jobs (window.jobData)
 *  - Render filters + apply filtering + compute match score + render cards
 *
 * Notes:
 *  - Industry filtering is “best effort”. Your DB often has null job_industries, so we:
 *      1) try fetching with industries (if present)
 *      2) fallback to fetching without industries if that returns 0
 *  - We always request a larger pool (limit=100) and then filter/dedupe client-side.
 */

(() => {
    // ----------------------------- Config ------------------------------
    const JOBS_ENDPOINT = "/.netlify/functions/get-jobs";
    const REQUEST_LIMIT = 100;           // pull enough rows to survive filtering/dedupe
    const DISPLAY_LIMIT = 50;            // max jobs shown
    const RECENT_DAYS = 90;              // recent window for display
    const JOBS_CACHE_KEY = "jobsCache:v1";
    const JOBS_CACHE_TTL_MS = 60 * 1000; // 1 min cache to reduce hammering
  
    // SKILL_DICT must exist globally (as in your current app).
    // If not, we’ll fallback to empty dict.
    const SKILL_DICT_SAFE = () => (typeof SKILL_DICT === "object" && SKILL_DICT) ? SKILL_DICT : {};
  
    // -------------------------- Small utils ---------------------------
    function escapeHtml(str) {
      return String(str ?? "").replace(/[&"'<>]/g, (s) => (
        { "&": "&amp;", '"': "&quot;", "'": "&#39;", "<": "&lt;", ">": "&gt;" }[s]
      ));
    }
  
    function fixEncoding(str) {
      if (!str) return "";
      return String(str)
        .replace(/â€™/g, "'")
        .replace(/â€“/g, "–")
        .replace(/â€”/g, "—")
        .replace(/â€œ/g, '"')
        .replace(/â€\x9d/g, '"')
        .replace(/â€˜/g, "'")
        .replace(/â€¢/g, "•")
        .replace(/Â/g, "")
        .replace(/â€¦/g, "…");
    }
  
    function asLowerString(v) {
      if (v == null) return "";
      if (Array.isArray(v)) return v.map(x => String(x ?? "").trim()).join(", ").toLowerCase();
      if (typeof v === "string") return v.toLowerCase();
      return String(v).toLowerCase();
    }
  
    function normFilterValue(v) {
      if (v == null) return "any";
      return String(v).trim().toLowerCase() || "any";
    }
  
    function normalizeIndustries(val) {
      if (Array.isArray(val)) return val.map(s => String(s).trim()).filter(Boolean);
      if (typeof val === "string") {
        const s = val.trim();
        if (!s) return [];
        // try JSON array
        try {
          const parsed = JSON.parse(s);
          if (Array.isArray(parsed)) return parsed.map(x => String(x).trim()).filter(Boolean);
        } catch (_) {}
        // fallback CSV-ish
        return s.split(/[;,|/]+/).map(x => x.trim()).filter(Boolean);
      }
      return [];
    }
  
    function safeUrl(url) {
      try {
        const s = String(url ?? "").trim();
        if (!s) return "#";
        if (/^https?:\/\//i.test(s)) return s;
        if (/^\//.test(s) || /^\.\//.test(s)) return s;
        return "#";
      } catch {
        return "#";
      }
    }
  
    function safeDateToISODate(raw) {
      if (!raw) return null;
  
      // Accept YYYY-MM-DD directly
      const s = String(raw).trim();
      if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
  
      // Try Date.parse
      const d = new Date(s);
      if (!isNaN(d)) return d.toISOString().slice(0, 10);
  
      // Attempt to extract YYYY-MM-DD from noisy strings
      const m = s.match(/(\d{4})-(\d{2})-(\d{2})/);
      if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  
      return null;
    }
  
    function timeAgo(dateString) {
      if (!dateString) return "";
      const date = new Date(dateString);
      if (isNaN(date)) return String(dateString);
  
      const now = new Date();
      const seconds = Math.floor((now - date) / 1000);
  
      let interval = Math.floor(seconds / 31536000);
      if (interval >= 1) return interval + " yr" + (interval === 1 ? "" : "s") + " ago";
  
      interval = Math.floor(seconds / 2592000);
      if (interval >= 1) return interval + " mo" + (interval === 1 ? "" : "s") + " ago";
  
      interval = Math.floor(seconds / 604800);
      if (interval >= 1) return interval + " wk" + (interval === 1 ? "" : "s") + " ago";
  
      interval = Math.floor(seconds / 86400);
      if (interval >= 1) return interval + " day" + (interval === 1 ? "" : "s") + " ago";
  
      interval = Math.floor(seconds / 3600);
      if (interval >= 1) return interval + " hr" + (interval === 1 ? "" : "s") + " ago";
  
      interval = Math.floor(seconds / 60);
      if (interval >= 1) return interval + " min" + (interval === 1 ? "" : "s") + " ago";
  
      return "Just now";
    }
  
    function showMessage(type, html) {
      const el = document.getElementById("dataMessage");
      if (!el) return;
      const klass =
        type === "success" ? "alert alert-success" :
        type === "danger"  ? "alert alert-danger" :
        type === "warning" ? "alert alert-warning" :
                             "alert alert-info";
      el.innerHTML = `<div class="${klass}" role="alert">${html}</div>`;
    }
  
    // ------------------------ Resume loading ---------------------------
    async function loadResumeData() {
      let data = localStorage.getItem("parsedResume");
  
      if (!data) {
        try {
          const resp = await fetch("./parsed_resume.json");
          if (resp.ok) {
            const json = await resp.json();
            data = JSON.stringify(json);
            showMessage("info", `Loaded parsed resume from <code>parsed_resume.json</code> (fallback).`);
          }
        } catch (e) {
          console.warn("No parsed resume in localStorage and failed to fetch parsed_resume.json", e);
        }
      }
  
      if (!data) {
        alert("No resume found. Please parse one first or provide a local parsed_resume.json file.");
        window.location.href = "index.html";
        return null;
      }
  
      const raw = JSON.parse(data);
      let resume = raw;
  
      if (raw && raw.inputs && Array.isArray(raw.inputs) && raw.inputs.length) {
        resume = raw.inputs[0];
  
        resume.name = resume.full_name || resume.name || "Unknown";
        resume.email = resume.email || resume.contact_email || "";
        resume.education = resume.education_level || resume.education || "Unknown";
        resume.years_experience = resume.years_experience || resume.years || 0;
        resume.title = resume.current_title || resume.title || "Unknown";
        resume.industries = resume.industries || resume.job_industries || [];
        resume.certifications = resume.certifications || [];
        resume.competitive_score = resume.competitive_score;
  
        if (!Array.isArray(resume.skills)) {
          if (typeof resume.skills === "string" && resume.skills.trim()) {
            resume.skills = resume.skills.split(/[,;|\n]+/).map(s => s.trim()).filter(Boolean);
          } else {
            resume.skills = [];
          }
        }
  
        resume.skills = normalizeSkills(resume.skills);
      } else {
        resume.name = resume.name || "Unknown";
        resume.email = resume.email || "";
        resume.education = resume.education || "Unknown";
        if (!Array.isArray(resume.skills)) {
          resume.skills = typeof resume.skills === "string"
            ? resume.skills.split(/[,;|\n]+/).map(s => s.trim()).filter(Boolean)
            : [];
        }
        resume.skills = normalizeSkills(resume.skills);
      }
  
      window.resumeData = resume;
      renderResumeSummary(resume);
  
      try {
        const insights = inferJobTypes(resume);
        populateJobTypeInsights(insights);
      } catch (e) {
        console.warn("Failed to infer job types", e);
      }
  
      return resume;
    }
  
    function normalizeSkills(skillsArr) {
      const skill_dict = SKILL_DICT_SAFE();
      const lowered = (skillsArr || []).map(s => String(s).toLowerCase());
      const out = new Set();
  
      lowered.forEach(s => {
        if (!s) return;
  
        // direct canonical match
        if (Object.prototype.hasOwnProperty.call(skill_dict, s)) {
          out.add(s);
          return;
        }
  
        // alias match
        for (const canonical of Object.keys(skill_dict)) {
          const aliases = skill_dict[canonical] || [];
          for (const a of aliases) {
            if (!a) continue;
            if (s.includes(String(a).toLowerCase())) {
              out.add(canonical);
              return;
            }
          }
        }
  
        out.add(s);
      });
  
      return Array.from(out);
    }
  
    function renderResumeSummary(resume) {
      const summaryEl = document.getElementById("resumeSummary");
      if (!summaryEl) return;
  
      const inds = Array.isArray(resume.industries) ? resume.industries.join(", ") : (resume.industries || "N/A");
      const certs = Array.isArray(resume.certifications) ? resume.certifications.join(", ") : (resume.certifications || "None");
      const skills = Array.isArray(resume.skills) ? resume.skills.join(", ") : "";
  
      summaryEl.innerHTML = `
        <div><b>Title:</b> ${escapeHtml(resume.title || "N/A")}</div>
        <div><b>Location:</b> ${escapeHtml(resume.location || "N/A")}</div>
        <div><b>Education:</b> ${escapeHtml(resume.education || "N/A")}</div>
        <div><b>Years Exp:</b> ${escapeHtml(resume.years_experience ?? 0)}</div>
        <div><b>Industries:</b> ${escapeHtml(inds)}</div>
        <div><b>Certifications:</b> ${escapeHtml(certs)}</div>
        <div><b>Competitive Score:</b> ${resume.competitive_score !== undefined ? escapeHtml(resume.competitive_score) : "N/A"}</div>
        <div><b>Skills:</b> ${escapeHtml(skills || "None detected")}</div>
      `;
    }
  
    // -------------------------- Job fetching ---------------------------
    function buildJobsUrl({ limit, industries }) {
      const url = new URL(JOBS_ENDPOINT, window.location.origin);
      url.searchParams.set("limit", String(limit));
      if (industries && industries.length) {
        url.searchParams.set("industries", industries.join(","));
      }
      return url.toString();
    }
  
    function getJobsCacheKey(url) {
      return `${JOBS_CACHE_KEY}:${url}`;
    }
  
    function readCache(url) {
      try {
        const key = getJobsCacheKey(url);
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || !parsed.ts || !Array.isArray(parsed.items)) return null;
        if (Date.now() - parsed.ts > JOBS_CACHE_TTL_MS) return null;
        return parsed.items;
      } catch {
        return null;
      }
    }
  
    function writeCache(url, items) {
      try {
        const key = getJobsCacheKey(url);
        localStorage.setItem(key, JSON.stringify({ ts: Date.now(), items }));
      } catch {}
    }
  
    async function fetchJobs(url) {
      const cached = readCache(url);
      if (cached) return cached;
  
      const resp = await fetch(url, { method: "GET", headers: { Accept: "application/json" } });
      const text = await resp.text();
  
      let json;
      try { json = JSON.parse(text); } catch {
        throw new Error(`Jobs API returned non-JSON (HTTP ${resp.status}): ${text.slice(0, 200)}`);
      }
  
      if (!resp.ok) {
        throw new Error(`Jobs API error (HTTP ${resp.status}): ${JSON.stringify(json).slice(0, 300)}`);
      }
  
      const items = Array.isArray(json) ? json : (json?.items || []);
      writeCache(url, items);
      return items;
    }
  
    function normalizeJob(item) {
      if (!item || typeof item !== "object" || Array.isArray(item)) return {};
  
      // Keep originals and add canonical fields
      const o = { ...item };
  
      o.job_title = o.job_title ?? o.title ?? o["job title"];
      o.company_name = o.company_name ?? o.company ?? o["company name"];
      o.job_location = o.job_location ?? o.location ?? o["job location"];
      o.job_summary = o.job_summary ?? o.description ?? o.summary ?? "";
      o.job_posted_date = o.job_posted_date ?? o.posted_date ?? o.posted_at ?? o.posted ?? "";
      o.job_seniority_level = o.job_seniority_level ?? o.seniority_level ?? "";
      o.job_employment_type = o.job_employment_type ?? o.employment_type ?? "";
      o.job_industries = o.job_industries ?? o.industries ?? null;
  
      // Many of your tables do not include apply_link/url; this makes UI robust.
      o.apply_link = o.apply_link ?? o.url ?? "";
  
      return o;
    }
  
    function uniqueByJobTitle(items) {
      const seen = new Set();
      const out = [];
      for (const row of items || []) {
        const key = String(row.job_title || "").trim().toLowerCase();
        if (!key) continue;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(row);
      }
      return out;
    }
  
    async function loadJobs(resume) {
      const industriesArray = Array.isArray(resume?.industries)
        ? resume.industries.filter(Boolean).map(String)
        : [];
  
      showMessage("info", "Loading jobs from live database...");
  
      let rawJobs = [];
      const urlWithIndustries = industriesArray.length
        ? buildJobsUrl({ limit: REQUEST_LIMIT, industries: industriesArray })
        : null;
  
      // 1) try with industries
      if (urlWithIndustries) {
        console.log("Fetching jobs (with industries):", urlWithIndustries);
        try {
          rawJobs = await fetchJobs(urlWithIndustries);
        } catch (e) {
          console.warn("Industry-filtered fetch failed, will fallback:", e);
        }
        console.log("Jobs returned (with industries):", rawJobs.length);
      }
  
      // 2) fallback to no industries if empty
      if (!rawJobs.length) {
        const urlNoIndustries = buildJobsUrl({ limit: REQUEST_LIMIT, industries: [] });
        console.log("Fetching jobs (no industries):", urlNoIndustries);
        rawJobs = await fetchJobs(urlNoIndustries);
        console.log("Jobs returned (no industries):", rawJobs.length);
      }
  
      const normalized = (rawJobs || []).map(normalizeJob).filter(j => j.job_title || j.company_name);
  
      window.jobData = normalized;
  
      showMessage("success", `Loaded <b>${window.jobData.length}</b> jobs from live database.`);
    }
  
    // -------------------------- Filters UI ----------------------------
    function populateFilters() {
      const companies = new Set();
      const titles = new Set();
      const locations = new Set();
      const industries = new Set();
      const seniorities = new Set();
      const employmentTypes = new Set();
  
      (window.jobData || []).forEach(job => {
        if (job.company_name) companies.add(String(job.company_name).trim());
        if (job.job_title) titles.add(String(job.job_title).trim());
        if (job.job_location) locations.add(String(job.job_location).trim());
  
        const inds = normalizeIndustries(job.job_industries);
        inds.forEach(i => industries.add(String(i).trim()));
  
        if (job.job_seniority_level) seniorities.add(String(job.job_seniority_level).trim());
        if (job.job_employment_type) employmentTypes.add(String(job.job_employment_type).trim());
      });
  
      function fill(id, set) {
        const sel = document.getElementById(id);
        if (!sel) return;
        sel.querySelectorAll('option:not([value="any"])').forEach(o => o.remove());
        Array.from(set).filter(Boolean).sort((a, b) => a.localeCompare(b)).forEach(val => {
          const opt = document.createElement("option");
          opt.value = String(val).toLowerCase();
          opt.text = val;
          sel.appendChild(opt);
        });
      }
  
      fill("filterCompany", companies);
      fill("filterTitle", titles);
      fill("filterLocation", locations);
      fill("filterIndustry", industries);
      fill("filterSeniority", seniorities);
      fill("filterEmployment", employmentTypes);
    }
  
    // --------------------------- Filtering ----------------------------
    function filterJobs() {
      const company   = document.getElementById("filterCompany") ? normFilterValue(document.getElementById("filterCompany").value) : "any";
      const title     = document.getElementById("filterTitle") ? normFilterValue(document.getElementById("filterTitle").value) : "any";
      const location  = document.getElementById("filterLocation") ? normFilterValue(document.getElementById("filterLocation").value) : "any";
      const industry  = document.getElementById("filterIndustry") ? normFilterValue(document.getElementById("filterIndustry").value) : "any";
      const seniority = document.getElementById("filterSeniority") ? normFilterValue(document.getElementById("filterSeniority").value) : "any";
      const employment= document.getElementById("filterEmployment") ? normFilterValue(document.getElementById("filterEmployment").value) : "any";
  
      const postedFrom = document.getElementById("postedFrom") ? document.getElementById("postedFrom").value : "";
      const postedTo   = document.getElementById("postedTo") ? document.getElementById("postedTo").value : "";
  
      const preferredRoles = (window.preferredJobRoles || []).map(s => asLowerString(s));
      const preferredEmployment = (window.preferredEmploymentTypes || []).map(s => asLowerString(s));
  
      const items = (window.jobData || []);
  
      let filtered = items.filter(job => {
        const jCompany    = asLowerString(job.company_name);
        const jTitle      = asLowerString(job.job_title);
        const jLocation   = asLowerString(job.job_location);
        const jIndustries = asLowerString(job.job_industries);
        const jSeniority  = asLowerString(job.job_seniority_level);
        const jEmployment = asLowerString(job.job_employment_type);
  
        const postedIso = safeDateToISODate(job.job_posted_date || job.posted_date || job.posted_at || "");
  
        const companyMatch    = (company === "any") ? true : jCompany.includes(company);
        const titleMatch      = (title === "any") ? true : jTitle.includes(title);
        const locationMatch   = (location === "any") ? true : jLocation.includes(location);
        const industryMatch   = (industry === "any") ? true : (jIndustries ? jIndustries.includes(industry) : false);
        const seniorityMatch  = (seniority === "any") ? true : jSeniority.includes(seniority);
        const employmentMatch = (employment === "any") ? true : jEmployment.includes(employment);
  
        let preferredRoleMatch = true;
        if (preferredRoles.length) {
          preferredRoleMatch = preferredRoles.some(r => r && (jTitle.includes(r) || jIndustries.includes(r)));
        }
  
        let preferredEmploymentMatch = true;
        if (preferredEmployment.length) {
          preferredEmploymentMatch = preferredEmployment.some(e => e && jEmployment.includes(e));
        }
  
        let postedMatch = true;
        if (postedFrom) postedMatch = postedMatch && (postedIso ? (postedIso >= postedFrom) : false);
        if (postedTo)   postedMatch = postedMatch && (postedIso ? (postedIso <= postedTo) : false);
  
        return companyMatch && titleMatch && locationMatch && industryMatch &&
               seniorityMatch && employmentMatch && postedMatch &&
               preferredRoleMatch && preferredEmploymentMatch;
      });
  
      const countBadge = document.getElementById("resultCountBadge");
      if (countBadge) countBadge.textContent = `${filtered.length} found`;
  
      const chartContainer = document.getElementById("matchChart")?.parentElement;
      if (chartContainer) {
        if (filtered.length > 0) chartContainer.classList.remove("d-none");
        else chartContainer.classList.add("d-none");
      }
  
      computeMatchAndRender(filtered);
    }
  
    // --------------------- Match scoring + render ----------------------
    function computeMatchAndRender(jobs) {
      const resumeSkills = (window.resumeData?.skills || []).map(s => String(s).toLowerCase());
      const skillDict = SKILL_DICT_SAFE();
  
      // Match score per job
      jobs.forEach(job => {
        const blob = (
          (job.job_summary || "") + " " +
          (job.job_description || "") + " " +
          (job.job_title || "") + " " +
          (Array.isArray(job.job_industries) ? job.job_industries.join(" ") : (job.job_industries || ""))
        ).toLowerCase();
  
        let matchedCount = 0;
        const matched = [];
  
        resumeSkills.forEach(skill => {
          const aliases = skillDict[skill] || [skill];
          const found = (aliases || []).find(a => a && blob.includes(String(a).toLowerCase()));
          if (found) {
            matchedCount += 1;
            matched.push(skill);
          }
        });
  
        job.match_score = resumeSkills.length ? (matchedCount / resumeSkills.length) * 100 : 0;
        job.matched_skills = matched;
      });
  
      // Sorting
      const sortBy = document.getElementById("sortBy")?.value || "relevance";
      if (sortBy === "score_desc" || sortBy === "relevance") {
        jobs.sort((a, b) => (b.match_score || 0) - (a.match_score || 0));
      } else if (sortBy === "score_asc") {
        jobs.sort((a, b) => (a.match_score || 0) - (b.match_score || 0));
      } else if (sortBy === "date_new") {
        jobs.sort((a, b) => {
          const da = new Date(a.job_posted_date || 0);
          const db = new Date(b.job_posted_date || 0);
          return db - da;
        });
      } else if (sortBy === "salary_high") {
        jobs.sort((a, b) => String(b.job_base_pay_range || "").localeCompare(String(a.job_base_pay_range || "")));
      }
  
      // Recent-only + cap
      const cutoff = new Date(Date.now() - RECENT_DAYS * 24 * 60 * 60 * 1000);
      const recent = jobs.filter(job => {
        const iso = safeDateToISODate(job.job_posted_date || job.posted_date || job.posted_at);
        if (!iso) return false;
        const d = new Date(iso);
        return !isNaN(d) && d >= cutoff;
      });
  
      const unique = uniqueByJobTitle(recent);
      const top = unique.slice(0, DISPLAY_LIMIT);
  
      renderJobCards(top);
      drawChart(top);
    }
  
    function renderJobCards(jobs) {
      const jobResults = document.getElementById("jobResults");
      if (!jobResults) return;
  
      if (!jobs.length) {
        jobResults.innerHTML =
          '<div class="text-center text-muted py-5"><i class="bi bi-inbox fs-1 d-block mb-3"></i>No jobs found matching your criteria.</div>';
        return;
      }
  
      jobResults.innerHTML = jobs.map(job => {
        const title = fixEncoding(job.job_title || "N/A");
        const company = fixEncoding(job.company_name || "N/A");
        const location = fixEncoding(job.job_location || "N/A");
        const jobType = fixEncoding(job.job_employment_type || "N/A");
        const seniority = fixEncoding(job.job_seniority_level || "N/A");
  
        const apply = safeUrl(job.apply_link || job.url);
  
        const safeTitle = escapeHtml(title);
        const safeCompany = escapeHtml(company);
        const safeLocation = escapeHtml(location);
        const safeJobType = escapeHtml(jobType);
        const safeSeniority = escapeHtml(seniority);
  
        const postedIso = safeDateToISODate(job.job_posted_date);
        const postedRelative = timeAgo(postedIso || job.job_posted_date);
  
        let summaryText = fixEncoding(job.job_summary || "");
        const MAX_SUMMARY_LEN = 550;
        if (summaryText.length > MAX_SUMMARY_LEN) summaryText = summaryText.slice(0, MAX_SUMMARY_LEN) + "...";
        const safeSummary = escapeHtml(summaryText);
  
        const salaryDisplay = job.job_base_pay_range ? escapeHtml(String(job.job_base_pay_range)) : "";
  
        let matchColor = "bg-secondary";
        if ((job.match_score || 0) >= 80) matchColor = "bg-success";
        else if ((job.match_score || 0) >= 50) matchColor = "bg-warning text-dark";
        else matchColor = "bg-danger";
  
        return `
          <div class="card job-card mb-3 p-3">
            <div class="row align-items-start">
              <div class="col-md-9">
                <h5 class="mb-1">
                  <a href="${apply}" target="_blank" class="job-title-link stretched-link-custom">${safeTitle}</a>
                </h5>
                <div class="mb-2">
                  <span class="company-name text-primary fw-bold">${safeCompany}</span>
                  <span class="text-muted mx-1">&middot;</span>
                  <span class="small text-muted"><i class="bi bi-geo-alt me-1"></i>${safeLocation}</span>
                </div>
  
                <div class="d-flex flex-wrap gap-3 mb-3 small text-muted">
                  ${jobType !== "N/A" ? `<span><i class="bi bi-briefcase me-1"></i>${safeJobType}</span>` : ""}
                  ${seniority !== "N/A" ? `<span><i class="bi bi-bar-chart me-1"></i>${safeSeniority}</span>` : ""}
                  ${salaryDisplay ? `<span class="fw-medium text-dark"><i class="bi bi-cash me-1"></i>${salaryDisplay}</span>` : ""}
                  <span class="text-success"><i class="bi bi-clock me-1"></i>${escapeHtml(postedRelative)}</span>
                </div>
              </div>
  
              <div class="col-md-3 text-end d-flex flex-column gap-2 align-items-end">
                <span class="badge ${matchColor} match-badge">
                  <i class="bi bi-stars me-1"></i>${Number(job.match_score || 0).toFixed(0)}% Match
                </span>
                <a href="${apply}" target="_blank" class="btn btn-outline-primary btn-sm w-100 mt-2">Apply Now</a>
              </div>
            </div>
  
            <div class="row mt-3">
              <div class="col-12">
                ${safeSummary ? `<div class="job-summary mb-2">${safeSummary}</div>` : ""}
              </div>
            </div>
          </div>
        `;
      }).join("");
    }
  
    // ---------------------------- Chart -------------------------------
    function drawChart(topJobs) {
      const canvas = document.getElementById("matchChart");
      if (!canvas || typeof Chart === "undefined") return;
      const ctx = canvas.getContext("2d");
      if (window.jobChart) window.jobChart.destroy();
  
      // No explicit colors per your prior style, but you used bootstrap blue; keep your behavior:
      window.jobChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: topJobs.map(j => j.job_title),
          datasets: [{
            label: "Match Score (%)",
            data: topJobs.map(j => j.match_score),
            backgroundColor: "#007bff",
          }],
        },
        options: {
          scales: { y: { beginAtZero: true, max: 100 } },
          plugins: { legend: { display: false } },
        },
      });
    }
  
    // ---------------- Job type inference + UI (kept) -------------------
    function inferJobTypes(resume) {
      const titleText = (resume.title || "").toLowerCase();
      const skillsText = Array.isArray(resume.skills) ? resume.skills.join(" ") : (resume.skills || "");
      const industriesText = Array.isArray(resume.industries) ? resume.industries.join(" ") : (resume.industries || "");
      const allText = (titleText + " " + skillsText + " " + industriesText).toLowerCase();
  
      const roleKeywords = {
        "Software Engineer": ["software", "engineer", "developer", "full stack", "backend", "frontend", "javascript", "java", "c#", "c++", "python", "ruby", "node", "react"],
        "Data / ML": ["data scientist", "data engineer", "data analyst", "machine learning", "ml", "spark", "pandas", "numpy", "scikit", "tensorflow", "pytorch", "etl", "big data"],
        "Product / PM": ["product manager", "product management", "pm", "product lead", "roadmap", "stakeholder"],
        "Management": ["manager", "lead", "supervisor", "director", "head of", "people manager"],
        "Education / Teaching": ["teacher", "education", "instructor", "professor", "tutor", "curriculum"],
        "Healthcare": ["nurse", "healthcare", "clinical", "medical", "caregiver", "patient"],
        "Sales": ["sales", "account executive", "business development", "bdm"],
        "Research": ["research", "researcher", "laboratory", "lab", "experimental"],
        "Intern / Entry Level": ["intern", "internship", "entry level", "graduate", "student"],
      };
  
      const roles = [];
  
      Object.keys(roleKeywords).forEach(role => {
        const kws = roleKeywords[role];
        let matches = 0;
        kws.forEach(k => { if (allText.includes(k.toLowerCase())) matches++; });
        if (matches > 0) {
          const score = Math.min(100, Math.round((matches / kws.length) * 100));
          roles.push({ name: role, score });
        }
      });
  
      roles.sort((a, b) => b.score - a.score);
  
      const employmentKeywords = {
        "Full-time": ["full time", "full-time", "fulltime"],
        "Part-time": ["part time", "part-time", "parttime"],
        "Contract": ["contract", "contractor"],
        "Temporary": ["temporary", "temp"],
        "Remote": ["remote", "work from home", "telecommute"],
      };
  
      const employment = [];
      Object.keys(employmentKeywords).forEach(e => {
        const kws = employmentKeywords[e];
        let matches = 0;
        kws.forEach(k => { if (allText.includes(k)) matches++; });
        if (matches > 0) {
          const score = Math.min(100, Math.round((matches / kws.length) * 100));
          employment.push({ name: e, score });
        }
      });
      employment.sort((a, b) => b.score - a.score);
  
      return { roles, employment };
    }
  
    function populateJobTypeInsights(insights) {
      const container = document.getElementById("jobTypeInsights");
      if (!container) return;
  
      const roles = insights.roles || [];
      const employment = insights.employment || [];
  
      const parts = [];
      parts.push('<div class="fw-bold">Suggested Job Types</div>');
  
      if (roles.length) {
        parts.push('<div class="mt-2"><small class="text-muted">Roles you may prefer:</small>');
        parts.push('<div class="d-flex flex-wrap gap-2 mt-1">');
        roles.forEach(r => {
          const safe = escapeHtml(r.name);
          const id = "role_" + safe.replace(/\s+/g, "_");
          parts.push(
            `<label class="form-check form-check-inline mb-1">
               <input class="form-check-input jobtype-role" type="checkbox" id="${id}" value="${safe}">
               <span class="form-check-label">${safe} <span class="badge bg-primary ms-1">${r.score}%</span></span>
             </label>`
          );
        });
        parts.push("</div></div>");
      }
  
      if (employment.length) {
        parts.push('<div class="mt-2"><small class="text-muted">Employment types in resume:</small>');
        parts.push('<div class="d-flex flex-wrap gap-2 mt-1">');
        employment.forEach(e => {
          const safe = escapeHtml(e.name);
          const id = "emp_" + safe.replace(/\s+/g, "_");
          parts.push(
            `<label class="form-check form-check-inline mb-1">
               <input class="form-check-input jobtype-emp" type="checkbox" id="${id}" value="${safe}">
               <span class="form-check-label">${safe} <span class="badge bg-secondary ms-1">${e.score}%</span></span>
             </label>`
          );
        });
        parts.push("</div></div>");
      }
  
      parts.push(
        '<div class="mt-2">' +
        '<button id="applyPreferredBtn" class="btn btn-sm btn-outline-primary me-2">Apply selected types</button>' +
        '<button id="clearPreferredBtn" class="btn btn-sm btn-outline-secondary">Clear</button>' +
        "</div>"
      );
  
      container.innerHTML = parts.join("");
  
      const apply = document.getElementById("applyPreferredBtn");
      const clear = document.getElementById("clearPreferredBtn");
  
      if (apply) apply.addEventListener("click", () => {
        const selRoles = Array.from(document.querySelectorAll(".jobtype-role:checked")).map(i => i.value);
        const selEmp = Array.from(document.querySelectorAll(".jobtype-emp:checked")).map(i => i.value);
        window.preferredJobRoles = selRoles;
        window.preferredEmploymentTypes = selEmp;
        filterJobs();
      });
  
      if (clear) clear.addEventListener("click", () => {
        Array.from(document.querySelectorAll(".jobtype-role:checked, .jobtype-emp:checked")).forEach(i => { i.checked = false; });
        window.preferredJobRoles = [];
        window.preferredEmploymentTypes = [];
        filterJobs();
      });
    }
  
    // --------------------------- Clear UI -----------------------------
    function clearFilters() {
      const ids = ["filterCompany", "filterTitle", "filterLocation", "filterIndustry", "filterSeniority", "filterEmployment"];
      ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = "any";
      });
      ["postedFrom", "postedTo", "minSalary", "maxSalary"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = "";
      });
      const sort = document.getElementById("sortBy");
      if (sort) sort.value = "relevance";
      filterJobs();
    }
  
    // expose clearFilters if your UI calls it
    window.clearFilters = clearFilters;
  
    // ----------------------------- Main -------------------------------
    async function main() {
      const resume = await loadResumeData();
      if (!resume) return;
  
      try {
        await loadJobs(resume);
      } catch (e) {
        console.error("Jobs load failed:", e);
        showMessage("danger", `<strong>Error loading jobs:</strong> ${escapeHtml(e.message)}<div class="small text-muted mt-2">Open DevTools → Console.</div>`);
        window.jobData = [];
      }
  
      populateFilters();
      filterJobs();
    }
  
    main();
  })();