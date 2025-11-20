async function loadResume() {
    const data = localStorage.getItem("parsedResume");
    if (!data) {
        alert("No resume found. Please parse one first.");
        window.location.href = "index.html";
        return;
    }
    const resume = JSON.parse(data);
    window.resumeData = resume;

    document.getElementById("resumeSummary").innerHTML = `
    <b>Name:</b> ${resume.name}<br>
    <b>Email:</b> ${resume.email}<br>
    <b>Education:</b> ${resume.education}<br>
    <b>Skills:</b> ${resume.skills.join(", ")}
  `;

    // Use prebuilt `jobs` array from data_as_arrays.js (required).
    // Support both patterns: `window.jobs` (var) and top-level `const jobs = [...]`.
    let sourceJobs = null;
    if (Array.isArray(window.jobs) && window.jobs.length) sourceJobs = window.jobs;
    else if (typeof jobs !== 'undefined' && Array.isArray(jobs) && jobs.length) sourceJobs = jobs;

    if (sourceJobs) {
        window.jobData = sourceJobs.map(item => {
            if (item && typeof item === 'object' && !Array.isArray(item)) {
                const o = {};
                Object.keys(item).forEach(k => { o[k.trim().toLowerCase()] = item[k]; });
                return o;
            }
            return item;
        });
        // show a small banner indicating which dataset is used
        const msgEl = document.getElementById('dataMessage');
        if (msgEl) {
            msgEl.innerHTML = `
                <div class="alert alert-info" role="alert">
                    Using dataset from <code>data_as_arrays.js</code> (loaded as <code>jobs</code>).
                </div>`;
        }
    } else {
        console.error('Prebuilt `jobs` array not found. Please include data_as_arrays.js that defines `jobs`.');
        window.jobData = [];
        const msgEl = document.getElementById('dataMessage');
        if (msgEl) {
            msgEl.innerHTML = `
                <div class="alert alert-warning" role="alert">
                    <strong>Jobs data not loaded:</strong> the prebuilt dataset <code>data_as_arrays.js</code> was not found or does not define <code>jobs</code>.
                    Please ensure the dataset file is included and defines <code>const jobs = [...]</code> or <code>window.jobs = [...]</code>.
                </div>`;
        }
    }
    populateFilters();
    filterJobs();
}

function csvToArray(str) {
    const [headerLine, ...lines] = str.trim().split("\n");
    const headers = headerLine.split(",").map(h => h.trim().toLowerCase());
    return lines.map(line => {
        const values = line.split(",");
        return headers.reduce((obj, h, i) => ({ ...obj, [h]: values[i] }), {});
    });
}

function populateFilters() {
    const companies = new Set();
    const titles = new Set();
    const locations = new Set();
    const industries = new Set();
    const seniorities = new Set();
    const employmentTypes = new Set();

    window.jobData.forEach(job => {
        if (job.company_name) companies.add(job.company_name.trim());
        if (job.job_title) titles.add(job.job_title.trim());
        if (job.job_location) locations.add(job.job_location.trim());
        if (job.job_industries) {
            job.job_industries.split(/[;,|\/]+/).forEach(i => {
                const v = i.trim();
                if (v) industries.add(v);
            });
        }
        if (job.job_seniority_level) seniorities.add(job.job_seniority_level.trim());
        if (job.job_employment_type) employmentTypes.add(job.job_employment_type.trim());
    });

    function fill(id, set) {
        const sel = document.getElementById(id);
        if (!sel) return;
        // remove existing options except the first 'any' option
        sel.querySelectorAll('option:not([value="any"])').forEach(o => o.remove());
        Array.from(set).sort((a, b) => a.localeCompare(b)).forEach(val => {
            const opt = document.createElement('option');
            opt.value = val.toLowerCase();
            opt.text = val;
            sel.appendChild(opt);
        });
    }

    fill('filterCompany', companies);
    fill('filterTitle', titles);
    fill('filterLocation', locations);
    fill('filterIndustry', industries);
    fill('filterSeniority', seniorities);
    fill('filterEmployment', employmentTypes);
}

function filterJobs() {
    const company = document.getElementById("filterCompany").value;
    const title = document.getElementById("filterTitle").value;
    const location = document.getElementById("filterLocation").value;
    const industry = document.getElementById("filterIndustry").value;
    const seniority = document.getElementById("filterSeniority") ? document.getElementById("filterSeniority").value : 'any';
    const employment = document.getElementById("filterEmployment") ? document.getElementById("filterEmployment").value : 'any';
    const postedFrom = document.getElementById("postedFrom") ? document.getElementById("postedFrom").value : '';
    const postedTo = document.getElementById("postedTo") ? document.getElementById("postedTo").value : '';

    let filtered = window.jobData.filter(job => {
        const jCompany = (job.company_name || "").toLowerCase();
        const jTitle = (job.job_title || "").toLowerCase();
        const jLocation = (job.job_location || "").toLowerCase();
        const jIndustries = (job.job_industries || "").toLowerCase();
        const jSeniority = (job.job_seniority_level || "").toLowerCase();
        const jEmployment = (job.job_employment_type || "").toLowerCase();
        const jPosted = job.job_posted_date || job.posted_date || job.posted || "";

        // normalize posted date to YYYY-MM-DD if possible
        let jPostedDate = null;
        if (jPosted) {
            const d = new Date(jPosted);
            if (!isNaN(d)) {
                // use ISO date string (YYYY-MM-DD)
                jPostedDate = d.toISOString().slice(0, 10);
            }
        }

        const companyMatch = (!company || company === 'any') ? true : jCompany.includes(company);
        const titleMatch = (!title || title === 'any') ? true : jTitle.includes(title);
        const locationMatch = (!location || location === 'any') ? true : jLocation.includes(location);
        const industryMatch = (!industry || industry === 'any') ? true : jIndustries.includes(industry);
        const seniorityMatch = (!seniority || seniority === 'any') ? true : jSeniority.includes(seniority);
        const employmentMatch = (!employment || employment === 'any') ? true : jEmployment.includes(employment);

        let postedMatch = true;
        if (postedFrom) {
            if (jPostedDate) postedMatch = postedMatch && (jPostedDate >= postedFrom);
            else postedMatch = false;
        }
        if (postedTo) {
            if (jPostedDate) postedMatch = postedMatch && (jPostedDate <= postedTo);
            else postedMatch = false;
        }

        return companyMatch && titleMatch && locationMatch && industryMatch && seniorityMatch && employmentMatch && postedMatch;
    });

    computeMatch(filtered);
}

function computeMatch(jobs) {
    const resumeSkills = window.resumeData.skills || [];
    jobs.forEach(job => {
        const text = (job.job_summary || "").toLowerCase();
        const matched = resumeSkills.filter(skill => text.includes(skill));
        job.match_score = matched.length ? (matched.length / resumeSkills.length) * 100 : 0;
    });

    jobs.sort((a, b) => b.match_score - a.match_score);
    const top = jobs.slice(0, 10);

    const jobResults = document.getElementById("jobResults");
    jobResults.innerHTML = top.map(job => `
    <div class="border p-3 mb-2 rounded bg-white">
      <b>${job.job_title}</b> at ${job.company_name} — ${job.job_location || "N/A"}<br>
      🧠 Match Score: <b>${job.match_score.toFixed(1)}%</b>
    </div>
  `).join("");

    drawChart(top);
}

function drawChart(topJobs) {
    const ctx = document.getElementById("matchChart").getContext("2d");
    if (window.jobChart) window.jobChart.destroy();

    window.jobChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topJobs.map(j => j.job_title),
            datasets: [{
                label: 'Match Score (%)',
                data: topJobs.map(j => j.match_score),
                backgroundColor: '#007bff'
            }]
        },
        options: {
            scales: { y: { beginAtZero: true, max: 100 } },
            plugins: { legend: { display: false } }
        }
    });
}

function clearFilters() {
    const ids = ['filterCompany', 'filterTitle', 'filterLocation', 'filterIndustry', 'filterSeniority', 'filterEmployment'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = 'any';
    });
    ['postedFrom', 'postedTo', 'minSalary', 'maxSalary'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    const sort = document.getElementById('sortBy');
    if (sort) sort.value = 'relevance';
    filterJobs();
}

loadResume();
