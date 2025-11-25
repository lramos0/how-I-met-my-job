async function loadResume() {
    let data = localStorage.getItem("parsedResume");

    // Loads local parsed_resume.json if localStorage is empty
    if (!data) {
        try {
            const resp = await fetch('./parsed_resume.json');
            if (resp.ok) {
                const json = await resp.json();
                data = JSON.stringify(json);
                const msgEl = document.getElementById('dataMessage');
                if (msgEl) {
                    msgEl.innerHTML = `
                        <div class="alert alert-info" role="alert">
                            Loaded parsed resume from <code>parsed_resume.json</code> (fallback).
                        </div>`;
                }
            }
        } catch (e) {
            // Ignores fetch errors
            console.warn('No parsed resume in localStorage and failed to fetch parsed_resume.json', e);
        }
    }

    if (!data) {
        alert("No resume found. Please parse one first or provide a local parsed_resume.json file.");
        window.location.href = "index.html";
        return;
    }

    // Supports two shapes:
    // 1) Legacy object: { name, email, education, skills: [] }
    // 2) Parser output with inputs: [{ ...fields... }]
    let raw = JSON.parse(data);
    let resume = raw;
    if (raw && raw.inputs && Array.isArray(raw.inputs) && raw.inputs.length) {
        // Uses first input record
        resume = raw.inputs[0];
        // Normalizes keys
        resume.name = resume.full_name || resume.name || "Unknown";
        resume.email = resume.email || resume.contact_email || "";
        resume.education = resume.education_level || resume.education || "Unknown";
        resume.years_experience = resume.years_experience || resume.years || 0;
        resume.title = resume.current_title || resume.title || "Unknown";
        resume.industries = resume.industries || resume.job_industries || [];
        resume.certifications = resume.certifications || [];
        resume.competitive_score = resume.competitive_score;
        // Ensures skills is an array
        if (!Array.isArray(resume.skills)) {
            if (typeof resume.skills === 'string' && resume.skills.trim()) {
                resume.skills = resume.skills.split(/[,;|\n]+/).map(s => s.trim()).filter(Boolean);
            } else {
                resume.skills = [];
            }
        }
        // Normalizes skills to canonical names
        const skill_dict = SKILL_DICT;

        function normalizeSkills(skillsArr) {
            const lowered = skillsArr.map(s => String(s).toLowerCase());
            const out = new Set();
            lowered.forEach(s => {
                // Direct match
                if (skill_dict[s]) {
                    out.add(s);
                    return;
                }
                // Checks aliases
                for (const canonical in skill_dict) {
                    const aliases = skill_dict[canonical];
                    for (let i = 0; i < aliases.length; i++) {
                        if (aliases[i] && s.indexOf(aliases[i]) !== -1) {
                            out.add(canonical);
                            return;
                        }
                    }
                }
                // Fallback: adds original token
                if (s) out.add(s);
            });
            return Array.from(out);
        }

        resume.skills = normalizeSkills(resume.skills);
    } else {
        // Normalizes legacy shape
        resume.name = resume.name || "Unknown";
        resume.email = resume.email || "";
        resume.education = resume.education || "Unknown";
        if (!Array.isArray(resume.skills)) {
            resume.skills = typeof resume.skills === 'string' ? resume.skills.split(/[,;|\n]+/).map(s => s.trim()).filter(Boolean) : [];
        }
    }

    window.resumeData = resume;

    // Shows resume summary
    const summaryEl = document.getElementById("resumeSummary");
    if (summaryEl) {
        summaryEl.innerHTML = `
        <div><b>Name:</b> ${resume.name}</div>
        <div><b>Email:</b> ${resume.email || 'N/A'}</div>
        <div><b>Title:</b> ${resume.title || 'N/A'}</div>
        <div><b>Location:</b> ${resume.location || 'N/A'}</div>
        <div><b>Education:</b> ${resume.education}</div>
        <div><b>Years Exp:</b> ${resume.years_experience || 0}</div>
        <div><b>Industries:</b> ${Array.isArray(resume.industries) ? resume.industries.join(', ') : (resume.industries || 'N/A')}</div>
        <div><b>Certifications:</b> ${Array.isArray(resume.certifications) ? resume.certifications.join(', ') : (resume.certifications || 'None')}</div>
        <div><b>Competitive Score:</b> ${resume.competitive_score !== undefined ? resume.competitive_score : 'N/A'}</div>
        <div><b>Skills:</b> ${(Array.isArray(resume.skills) ? resume.skills.join(', ') : '') || 'None detected'}</div>
      `;
    }

    // Infer job-type insights and render them
    try {
        const insights = inferJobTypes(resume);
        populateJobTypeInsights(insights);
    } catch (e) {
        console.warn('Failed to infer job types', e);
    }

    // Fetches jobs
    try {
        const response = await fetch("/.netlify/functions/get-jobs");
        if (!response.ok) throw new Error("Failed to load jobs");
        const jobsData = await response.json();

        window.jobData = jobsData.map(item => {
            // Normalizes keys and handles schema differences
            const o = {};
            if (item && typeof item === 'object' && !Array.isArray(item)) {
                Object.keys(item).forEach(k => {
                    o[k.trim().toLowerCase()] = item[k];
                });

                // Populates job_employment_type
                if (!o.job_employment_type && o.employment_type) {
                    o.job_employment_type = o.employment_type;
                }
                // Populates job_seniority_level
                if (!o.job_seniority_level && o.seniority_level) {
                    o.job_seniority_level = o.seniority_level;
                }
                // Populates job_posted_date
                if (!o.job_posted_date && o.posted_date) {
                    o.job_posted_date = o.posted_date;
                }
            }
            return o;
        });

        const msgEl = document.getElementById('dataMessage');
        if (msgEl) {
            msgEl.innerHTML = `
                <div class="alert alert-success" role="alert">
                    Loaded ${window.jobData.length} jobs from live database.
                </div>`;
        }
    } catch (err) {
        console.error(err);
        window.jobData = [];
        const msgEl = document.getElementById('dataMessage');
        if (msgEl) {
            msgEl.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <strong>Error loading jobs:</strong> ${err.message}.
                </div>`;
        }
    }
    populateFilters();
    filterJobs();
}

// Skill dictionary
const SKILL_DICT = {
    "python": ["python", "python3", "cpython", "pypy"],
    "java": ["java", "openjdk", "jdk", "jvm"],
    "javascript": ["javascript", "js", "nodejs", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "go": ["go", "golang"],
    "c": ["c"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "ruby": ["ruby", "ruby on rails", "rails"],
    "php": ["php", "laravel", "symfony"],
    "rust": ["rust"],
    "kotlin": ["kotlin"],
    "scala": ["scala"],
    "r": ["r", "r language"],
    "swift": ["swift"],
    "shell": ["shell", "bash", "zsh", "sh"],
    "perl": ["perl"],
    "sql": ["sql", "mysql", "postgresql", "oracle", "sqlite", "mssql"],
    "nosql": ["nosql", "mongodb", "cassandra", "redis", "dynamodb", "couchdb"],
    "big data": ["big data", "hadoop", "spark", "mapreduce", "hive", "pig"],
    "data engineering": ["data engineering", "etl", "data pipeline", "airflow"],
    "data science": ["data science", "machine learning", "ml", "statistics"],
    "deep learning": ["deep learning", "neural networks", "tensorflow", "pytorch", "keras"],
    "mlops": ["mlops", "model deployment", "model serving", "sagemaker", "mlflow"],
    "nlp": ["nlp", "natural language processing", "transformers", "spacy", "nltk"],
    "computer vision": ["computer vision", "cv", "opencv"],
    "analytics": ["analytics", "business intelligence", "bi", "tableau", "power bi"],
    "rest": ["rest", "restful api", "api", "web api"],
    "graphql": ["graphql"],
    "web frameworks": ["django", "flask", "express", "spring", "rails", "fastapi"],
    "frontend frameworks": ["react", "angular", "vue", "svelte", "ember"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "scss", "sass", "less"],
    "webpack": ["webpack", "rollup", "parcel"],
    "microservices": ["microservices", "service oriented architecture", "soa"],
    "devops": ["devops", "ci/cd", "continuous integration", "continuous delivery", "continuous deployment"],
    "docker": ["docker", "containers"],
    "kubernetes": ["kubernetes", "k8s", "kube"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "chef": ["chef"],
    "puppet": ["puppet"],
    "helm": ["helm"],
    "istio": ["istio", "service mesh"],
    "prometheus": ["prometheus", "grafana", "monitoring"],
    "logging": ["elk", "elasticsearch", "logstash", "kibana", "splunk"],
    "cloud aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation", "iam", "dynamodb"],
    "cloud azure": ["azure", "microsoft azure", "azure functions", "azure devops", "arm templates"],
    "cloud gcp": ["gcp", "google cloud", "google cloud platform", "gce", "bigquery", "cloud functions"],
    "openstack": ["openstack"],
    "serverless": ["serverless", "faas"],
    "edge computing": ["edge computing"],
    "networking": ["networking", "dns", "http", "tcp/ip"],
    "security": ["security", "kubernetes security", "oauth2", "jwt", "tls", "ssl", "vault"],
    "git": ["git", "gitlab", "github", "bitbucket"],
    "ci tools": ["jenkins", "circleci", "travis ci", "github actions", "gitlab ci", "azure pipelines"],
    "jira": ["jira", "confluence"],
    "slack": ["slack"],
    "docker-compose": ["docker-compose", "compose"],
    "testing": ["testing", "unit test", "integration test", "pytest", "junit", "mocha", "jest"],
    "performance": ["performance", "profiling", "benchmark"],
    "cache": ["cache", "redis", "memcached"],
    "microservices architecture": ["microservices architecture", "soa", "service mesh"],
    "design patterns": ["design patterns", "solid", "ddd", "clean architecture"],
    "architecture": ["architecture", "system design", "scalability", "high availability"],
    "agile": ["agile", "scrum", "kanban"],
    "devsecops": ["devsecops", "security as code", "shift left"],
    "observability": ["observability", "opentelemetry", "logging", "tracing", "metrics"]
};

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
        // Removes options except 'any'
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
    const company = document.getElementById("filterCompany") ? document.getElementById("filterCompany").value : 'any';
    const title = document.getElementById("filterTitle") ? document.getElementById("filterTitle").value : 'any';
    const location = document.getElementById("filterLocation") ? document.getElementById("filterLocation").value : 'any';
    const industry = document.getElementById("filterIndustry") ? document.getElementById("filterIndustry").value : 'any';
    const seniority = document.getElementById("filterSeniority") ? document.getElementById("filterSeniority").value : 'any';
    const employment = document.getElementById("filterEmployment") ? document.getElementById("filterEmployment").value : 'any';
    const postedFrom = document.getElementById("postedFrom") ? document.getElementById("postedFrom").value : '';
    const postedTo = document.getElementById("postedTo") ? document.getElementById("postedTo").value : '';

    // Preferred filters (optional)
    const preferredRoles = (window.preferredJobRoles || []).map(s => s.toLowerCase());
    const preferredEmployment = (window.preferredEmploymentTypes || []).map(s => s.toLowerCase());

    let filtered = window.jobData.filter(job => {
        const jCompany = (job.company_name || "").toLowerCase();
        const jTitle = (job.job_title || "").toLowerCase();
        const jLocation = (job.job_location || "").toLowerCase();
        const jIndustries = (job.job_industries || "").toLowerCase();
        const jSeniority = (job.job_seniority_level || "").toLowerCase();
        const jEmployment = (job.job_employment_type || "").toLowerCase();
        const jPosted = job.job_posted_date || job.posted_date || job.posted || "";

        // Normalizes posted date
        let jPostedDate = null;
        if (jPosted) {
            const d = new Date(jPosted);
            if (!isNaN(d)) {
                // Uses ISO date
                jPostedDate = d.toISOString().slice(0, 10);
            }
        }

        const companyMatch = (!company || company === 'any') ? true : jCompany.includes(company);
        const titleMatch = (!title || title === 'any') ? true : jTitle.includes(title);
        const locationMatch = (!location || location === 'any') ? true : jLocation.includes(location);
        const industryMatch = (!industry || industry === 'any') ? true : jIndustries.includes(industry);
        const seniorityMatch = (!seniority || seniority === 'any') ? true : jSeniority.includes(seniority);
        const employmentMatch = (!employment || employment === 'any') ? true : jEmployment.includes(employment);

        // Checks preferred roles
        let preferredRoleMatch = true;
        if (preferredRoles.length) {
            preferredRoleMatch = preferredRoles.some(r => jTitle.includes(r) || jIndustries.includes(r));
        }

        // Checks preferred employment types
        let preferredEmploymentMatch = true;
        if (preferredEmployment.length) {
            preferredEmploymentMatch = preferredEmployment.some(e => jEmployment.includes(e));
        }

        let postedMatch = true;
        if (postedFrom) {
            if (jPostedDate) postedMatch = postedMatch && (jPostedDate >= postedFrom);
            else postedMatch = false;
        }
        if (postedTo) {
            if (jPostedDate) postedMatch = postedMatch && (jPostedDate <= postedTo);
            else postedMatch = false;
        }

        return companyMatch && titleMatch && locationMatch && industryMatch && seniorityMatch && employmentMatch && postedMatch && preferredRoleMatch && preferredEmploymentMatch;
    });

    computeMatch(filtered);
}

function timeAgo(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    if (isNaN(date)) return dateString; // Fallback for invalid date

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

function computeMatch(jobs) {
    const resumeSkills = window.resumeData.skills || [];
    jobs.forEach(job => {
        const text = ((job.job_summary || "") + " " + (job.job_description || "") + " " + (job.job_title || "") + " " + (job.job_industries || "")).toLowerCase();
        let matchedCount = 0;
        const matched = [];

        resumeSkills.forEach(skill => {
            const aliases = SKILL_DICT[skill] || [skill];
            const foundAlias = aliases.find(a => a && text.indexOf(a) !== -1);
            if (foundAlias) {
                matchedCount += 1;
                matched.push(skill);
            }
        });

        job.match_score = resumeSkills.length ? (matchedCount / resumeSkills.length) * 100 : 0;
        job.matched_skills = matched;
    });

    jobs.sort((a, b) => b.match_score - a.match_score);

    // Filter by last 30days
    const days_old = 90;
    const cutoff_date = new Date(Date.now() - days_old * 24 *60 * 60 * 1000);
    const recent = jobs.filter(job => {
        const raw = job.job_posted_date || job.posted_date; // be robust
        if (!raw) return false;
        const d = new Date(raw);
        return !isNaN(d) && d >= cutoff_date;
    });
    const top = recent.slice(0, 50); // 
    

    const jobResults = document.getElementById("jobResults");
    if (jobResults) {
        jobResults.innerHTML = top.map(job => {
            const title = job.job_title || 'N/A';
            const company = job.company_name || 'N/A';
            const location = job.job_location || 'N/A';
            const jobType = job.job_employment_type || 'N/A';
            const seniority = job.job_seniority_level || 'N/A';
            const applyLink = job.job_url || '#';

            // Parses posted date
            let postedRaw = job.job_posted_date;
            const postedRelative = timeAgo(postedRaw);

            const salary = job.salary || '';

            // Formats salary
            let salaryDisplay = '';
            if (typeof salary === 'string' && salary.trim()) salaryDisplay = salary;
            else if (typeof salary === 'object' && salary !== null) {
                const min = salary.min || salary.min_salary || salary.salary_min;
                const max = salary.max || salary.max_salary || salary.salary_max;
                if (min || max) salaryDisplay = `$${min || '?'} - $${max || '?'} `;
            }

            // Match score color
            let matchColor = 'bg-secondary';
            if (job.match_score >= 70) matchColor = 'bg-success';
            else if (job.match_score >= 40) matchColor = 'bg-warning text-dark';

            return `
            <div class="card job-card mb-3 p-3">
                <div class="row g-0">
                    <div class="col-md-10">
                        <h5 class="mb-1">
                            <a href="${applyLink}" target="_blank" class="job-title-link stretched-link-custom">${title}</a>
                        </h5>
                        <div class="mb-2">
                            <span class="company-name">${company}</span>
                            <span class="metadata-text mx-1">•</span>
                            <span class="metadata-text">${location}</span>
                        </div>
                        
                        <div class="mb-2 metadata-text">
                            ${jobType !== 'N/A' ? `<span class="me-3"><i class="bi bi-briefcase-fill me-1"></i>${jobType}</span>` : ''}
                            ${seniority !== 'N/A' ? `<span class="me-3"><i class="bi bi-bar-chart-fill me-1"></i>${seniority}</span>` : ''}
                            ${salaryDisplay ? `<span class="me-3"><i class="bi bi-cash me-1"></i>${salaryDisplay}</span>` : ''}
                            <span class="text-success"><i class="bi bi-clock-history me-1"></i>${postedRelative}</span>
                        </div>

                        ${job.job_summary ? `<div class="job-summary text-muted mt-2">${job.job_summary}</div>` : ''}
                    </div>
                    
                    <div class="col-md-2 d-flex flex-column align-items-end justify-content-between">
                        <span class="badge ${matchColor} match-badge p-2 rounded-pill">
                            ${job.match_score.toFixed(0)}% Match
                        </span>
                        <div class="mt-3">
                             <a href="${applyLink}" target="_blank" class="btn btn-apply btn-sm text-decoration-none">Apply</a>
                        </div>
                    </div>
                </div>
            </div>
        `;
        }).join("");
    }

    // Initializes apply modal
    initApplyModal();

    drawChart(top);
}

///////////////// Job type inference + UI ////////////////////////////////
function inferJobTypes(resume) {
    // Builds text blob
    const titleText = (resume.title || '').toLowerCase();
    const skillsText = Array.isArray(resume.skills) ? resume.skills.join(' ') : (resume.skills || '');
    const industriesText = Array.isArray(resume.industries) ? resume.industries.join(' ') : (resume.industries || '');
    const allText = (titleText + ' ' + skillsText + ' ' + industriesText).toLowerCase();

    // Role keywords
    const roleKeywords = {
        'Software Engineer': ['software', 'engineer', 'developer', 'full stack', 'full-stack', 'backend', 'frontend', 'javascript', 'java', 'c#', 'c++', 'python', 'ruby', 'node', 'react'],
        'Data / ML': ['data scientist', 'data engineer', 'data analyst', 'machine learning', 'ml', 'spark', 'pandas', 'numpy', 'scikit', 'tensorflow', 'pytorch', 'etl', 'big data'],
        'Product / PM': ['product manager', 'product management', '\bpm\b', 'product lead', 'roadmap', 'stakeholder'],
        'Management': ['manager', 'lead', 'supervisor', 'director', 'head of', 'people manager'],
        'Education / Teaching': ['teacher', 'education', 'instructor', 'professor', 'tutor', 'curriculum'],
        'Healthcare': ['nurse', 'healthcare', 'clinical', 'medical', 'caregiver', 'patient'],
        'Sales': ['sales', 'account executive', 'business development', 'bdm'],
        'Research': ['research', 'researcher', 'laboratory', 'lab', 'experimental'],
        'Intern / Entry Level': ['intern', 'internship', 'entry level', 'graduate', 'student']
    };

    // Industry to role mapping
    const industryRoleMap = {
        'software': ['Software Engineer'],
        'technology': ['Software Engineer', 'Data / ML'],
        'education': ['Education / Teaching'],
        'health': ['Healthcare'],
        'finance': ['Data / ML', 'Finance'],
        'research': ['Research']
    };

    const roles = [];

    // Scores roles by keyword matches
    Object.keys(roleKeywords).forEach(role => {
        const kws = roleKeywords[role];
        let matches = 0;
        kws.forEach(k => {
            try {
                const re = new RegExp(k.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'), 'gi');
                if (allText.match(re)) matches += (allText.match(re) || []).length;
            } catch (e) {
                // Fallback to indexOf
                if (allText.indexOf(k.toLowerCase()) !== -1) matches++;
            }
        });
        if (matches > 0) {
            // Calculates confidence score
            const score = Math.min(100, Math.round((matches / kws.length) * 100));
            roles.push({ name: role, score });
        }
    });

    // Adds roles from industry
    if (roles.length === 0 && industriesText) {
        Object.keys(industryRoleMap).forEach(ind => {
            if (industriesText.includes(ind)) {
                industryRoleMap[ind].forEach(r => {
                    // Adds with low confidence
                    roles.push({ name: r, score: 40 });
                });
            }
        });
    }

    // Employment types scoring
    const employmentKeywords = {
        'Full-time': ['full time', 'full-time', 'fulltime'],
        'Part-time': ['part time', 'part-time', 'parttime'],
        'Contract': ['contract', 'contractor'],
        'Temporary': ['temporary', '\btemp\b'],
        'Remote': ['remote', 'work from home', 'telecommute']
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

    // Sorts roles
    roles.sort((a, b) => b.score - a.score);
    employment.sort((a, b) => b.score - a.score);

    return { roles, employment };
}

function populateJobTypeInsights(insights) {
    const container = document.getElementById('jobTypeInsights');
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
            const id = 'role_' + safe.replace(/\s+/g, '_');
            parts.push(`<label class="form-check form-check-inline mb-1"><input class="form-check-input jobtype-role" type="checkbox" id="${id}" value="${safe}"> <span class="form-check-label">${safe} <span class="badge bg-primary ms-1">${r.score}%</span></span></label>`);
        });
        parts.push('</div></div>');
    }

    if (employment.length) {
        parts.push('<div class="mt-2"><small class="text-muted">Employment types in resume:</small>');
        parts.push('<div class="d-flex flex-wrap gap-2 mt-1">');
        employment.forEach(e => {
            const safe = escapeHtml(e.name);
            const id = 'emp_' + safe.replace(/\s+/g, '_');
            parts.push(`<label class="form-check form-check-inline mb-1"><input class="form-check-input jobtype-emp" type="checkbox" id="${id}" value="${safe}"> <span class="form-check-label">${safe} <span class="badge bg-secondary ms-1">${e.score}%</span></span></label>`);
        });
        parts.push('</div></div>');
    }

    parts.push('<div class="mt-2"><button id="applyPreferredBtn" class="btn btn-sm btn-outline-primary me-2">Apply selected types</button><button id="clearPreferredBtn" class="btn btn-sm btn-outline-secondary">Clear</button></div>');

    container.innerHTML = parts.join('');

    // Wires buttons
    const apply = document.getElementById('applyPreferredBtn');
    const clear = document.getElementById('clearPreferredBtn');
    if (apply) apply.addEventListener('click', () => {
        const selRoles = Array.from(document.querySelectorAll('.jobtype-role:checked')).map(i => i.value);
        const selEmp = Array.from(document.querySelectorAll('.jobtype-emp:checked')).map(i => i.value);
        window.preferredJobRoles = selRoles;
        window.preferredEmploymentTypes = selEmp;
        filterJobs();
    });
    if (clear) clear.addEventListener('click', () => {
        Array.from(document.querySelectorAll('.jobtype-role:checked, .jobtype-emp:checked')).forEach(i => i.checked = false);
        window.preferredJobRoles = [];
        window.preferredEmploymentTypes = [];
        filterJobs();
    });
}

// Escapes HTML
function escapeHtml(str) {
    return String(str).replace(/[&"'<>]/g, function (s) {
        return ({ '&': '&amp;', '"': '&quot;', "'": '&#39;', '<': '&lt;', '>': '&gt;' }[s]);
    });
}

// Apply modal handling
function showApplyModal(link, title, company) {
    const modal = document.getElementById('applyModal');
    if (!modal) return;
    const body = document.getElementById('applyModalBody');
    const heading = document.getElementById('applyModalTitle');
    heading.textContent = `Open application for: ${title} @ ${company}`;
    body.textContent = 'This will open the application page in a new tab. Continue?';
    modal.dataset.link = link;
    modal.style.display = 'flex';
}

function hideApplyModal() {
    const modal = document.getElementById('applyModal');
    if (!modal) return;
    modal.style.display = 'none';
    delete modal.dataset.link;
}

function initApplyModal() {
    const jobResults = document.getElementById('jobResults');
    if (jobResults) {
        jobResults.addEventListener('click', (e) => {
            const btn = e.target.closest('.apply-btn');
            if (!btn) return;
            const link = btn.getAttribute('data-link') || '#';
            const title = btn.getAttribute('data-title') || 'Job';
            const company = btn.getAttribute('data-company') || '';
            showApplyModal(link, title, company);
        });
    }

    const cancel = document.getElementById('applyCancelBtn');
    const confirm = document.getElementById('applyConfirmBtn');
    const modal = document.getElementById('applyModal');
    if (cancel) cancel.addEventListener('click', hideApplyModal);
    if (confirm) confirm.addEventListener('click', () => {
        if (!modal) return;
        const link = modal.dataset.link;
        if (link) {
            try {
                window.open(link, '_blank', 'noopener');
            } catch (e) {
                // Fallback
                const a = document.createElement('a');
                a.href = link;
                a.target = '_blank';
                a.rel = 'noopener';
                a.click();
            }
        }
        hideApplyModal();
    });
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
