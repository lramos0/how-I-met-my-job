async function loadResume() {
    let data = localStorage.getItem("parsedResume");

    // If not in localStorage, try to load a local `parsed_resume.json` file (useful for testing).
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
            // ignore fetch errors and fall through to alert below
            console.warn('No parsed resume in localStorage and failed to fetch parsed_resume.json', e);
        }
    }

    if (!data) {
        alert("No resume found. Please parse one first or provide a local parsed_resume.json file.");
        window.location.href = "index.html";
        return;
    }

    // Support two shapes:
    // 1) Lightweight object saved as `parsedResume` (legacy): { name, email, education, skills: [] }
    // 2) Parser output with `inputs: [{ ...fields... }]` produced by `index.html` parser
    let raw = JSON.parse(data);
    let resume = raw;
    if (raw && raw.inputs && Array.isArray(raw.inputs) && raw.inputs.length) {
        // prefer the first input record
        resume = raw.inputs[0];
        // normalize common keys for downstream code
        resume.name = resume.full_name || resume.name || "Unknown";
        resume.email = resume.email || resume.contact_email || "";
        resume.education = resume.education_level || resume.education || "Unknown";
        resume.years_experience = resume.years_experience || resume.years || 0;
        resume.title = resume.current_title || resume.title || "Unknown";
        resume.industries = resume.industries || resume.job_industries || [];
        resume.certifications = resume.certifications || [];
        resume.competitive_score = resume.competitive_score;
        // ensure skills is an array
        if (!Array.isArray(resume.skills)) {
            if (typeof resume.skills === 'string' && resume.skills.trim()) {
                resume.skills = resume.skills.split(/[,;|\n]+/).map(s => s.trim()).filter(Boolean);
            } else {
                resume.skills = [];
            }
        }
        // Normalize skills to canonical names using the same skill dictionary as the parser
        const skill_dict = {
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

        function normalizeSkills(skillsArr) {
            const lowered = skillsArr.map(s => String(s).toLowerCase());
            const out = new Set();
            lowered.forEach(s => {
                // direct canonical match
                if (skill_dict[s]) {
                    out.add(s);
                    return;
                }
                // check aliases
                for (const canonical in skill_dict) {
                    const aliases = skill_dict[canonical];
                    for (let i = 0; i < aliases.length; i++) {
                        if (aliases[i] && s.indexOf(aliases[i]) !== -1) {
                            out.add(canonical);
                            return;
                        }
                    }
                }
                // fallback: add the original token
                if (s) out.add(s);
            });
            return Array.from(out);
        }

        resume.skills = normalizeSkills(resume.skills);
    } else {
        // normalize the legacy shape
        resume.name = resume.name || "Unknown";
        resume.email = resume.email || "";
        resume.education = resume.education || "Unknown";
        if (!Array.isArray(resume.skills)) {
            resume.skills = typeof resume.skills === 'string' ? resume.skills.split(/[,;|\n]+/).map(s => s.trim()).filter(Boolean) : [];
        }
    }

    window.resumeData = resume;

    // Show a richer resume summary on the jobs page
    document.getElementById("resumeSummary").innerHTML = `
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

    // Infer job-type insights and render them
    // Infer job-type insights and render them
    try {
        const insights = inferJobTypes(resume);
        populateJobTypeInsights(insights);
    } catch (e) {
        console.warn('Failed to infer job types', e);
    }

    // Fetch jobs from backend
    try {
        const response = await fetch("/.netlify/functions/get-jobs");
        if (!response.ok) throw new Error("Failed to load jobs");
        const jobsData = await response.json();

        window.jobData = jobsData.map(item => {
            // Normalize keys to lowercase and handle Supabase/Firestore schema differences
            const o = {};
            if (item && typeof item === 'object' && !Array.isArray(item)) {
                Object.keys(item).forEach(k => {
                    o[k.trim().toLowerCase()] = item[k];
                });

                // Ensure job_employment_type is populated if it comes as employment_type
                if (!o.job_employment_type && o.employment_type) {
                    o.job_employment_type = o.employment_type;
                }
                // Ensure job_seniority_level is populated if it comes as seniority_level
                if (!o.job_seniority_level && o.seniority_level) {
                    o.job_seniority_level = o.seniority_level;
                }
                // Ensure job_posted_date is populated if it comes as posted_date
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

// Global skill dictionary reused for matching (keeps parity with parser)
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

    // Preferred filters set by insights UI (optional)
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

        // If preferred roles are set, require at least one role to appear in title or industries
        let preferredRoleMatch = true;
        if (preferredRoles.length) {
            preferredRoleMatch = preferredRoles.some(r => jTitle.includes(r) || jIndustries.includes(r));
        }

        // If preferred employment types are set, require job employment to match any
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
    if (isNaN(date)) return dateString; // Fallback if not a valid date

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
    const top = jobs.slice(0, 50); // Limit to 50 as requested

    const jobResults = document.getElementById("jobResults");
    jobResults.innerHTML = top.map(job => {
        const title = job.job_title || job.title || 'N/A';
        const company = job.company_name || job.company || 'N/A';
        const location = job.job_location || job.location || 'N/A';
        const jobType = job.job_employment_type || job.employment_type || job.type || 'N/A';
        const seniority = job.job_seniority_level || job.seniority_level || 'N/A';
        const applyLink = job.job_url || job.url || job.apply_link || job.application_link || '#';

        // Handle posted date parsing
        let postedRaw = job.job_posted_date || job.posted_date || job.posted;
        // If it's an object with seconds (Firestore style) or just needs standardizing
        if (postedRaw && typeof postedRaw === 'object' && postedRaw.seconds) {
            postedRaw = new Date(postedRaw.seconds * 1000).toISOString();
        }
        const postedRelative = timeAgo(postedRaw);

        const salary = job.salary || job.pay || job.compensation || job.salary_range || job.pay_range || '';

        // format salary display if object-like (min/max)
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

    // initialize apply modal handlers for the newly-rendered buttons
    initApplyModal();

    drawChart(top);
}

/* --------------------
   Job type inference + UI
   -------------------- */
function inferJobTypes(resume) {
    // Build a normalized text blob from title, skills and industries
    const titleText = (resume.title || '').toLowerCase();
    const skillsText = Array.isArray(resume.skills) ? resume.skills.join(' ') : (resume.skills || '');
    const industriesText = Array.isArray(resume.industries) ? resume.industries.join(' ') : (resume.industries || '');
    const allText = (titleText + ' ' + skillsText + ' ' + industriesText).toLowerCase();

    // Define keyword sets per role for better precision
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

    // Map industries to suggested roles (soft mapping)
    const industryRoleMap = {
        'software': ['Software Engineer'],
        'technology': ['Software Engineer', 'Data / ML'],
        'education': ['Education / Teaching'],
        'health': ['Healthcare'],
        'finance': ['Data / ML', 'Finance'],
        'research': ['Research']
    };

    const roles = [];

    // Score each role by counting keyword matches (case-insensitive)
    Object.keys(roleKeywords).forEach(role => {
        const kws = roleKeywords[role];
        let matches = 0;
        kws.forEach(k => {
            try {
                const re = new RegExp(k.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'), 'gi');
                if (allText.match(re)) matches += (allText.match(re) || []).length;
            } catch (e) {
                // if regex creation fails fallback to indexOf
                if (allText.indexOf(k.toLowerCase()) !== -1) matches++;
            }
        });
        if (matches > 0) {
            // confidence: matched keywords / total keywords (capped at 1) scaled to 0-100
            const score = Math.min(100, Math.round((matches / kws.length) * 100));
            roles.push({ name: role, score });
        }
    });

    // Augment roles from industry names when role not present
    if (roles.length === 0 && industriesText) {
        Object.keys(industryRoleMap).forEach(ind => {
            if (industriesText.includes(ind)) {
                industryRoleMap[ind].forEach(r => {
                    // add with low confidence (40)
                    roles.push({ name: r, score: 40 });
                });
            }
        });
    }

    // Employment types with keyword sets and scoring
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

    // Sort roles by confidence desc
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

    // wire buttons
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

// Simple helper to escape HTML in attributes
function escapeHtml(str) {
    return String(str).replace(/[&"'<>]/g, function (s) {
        return ({ '&': '&amp;', '"': '&quot;', "'": '&#39;', '<': '&lt;', '>': '&gt;' }[s]);
    });
}

// Modal handling for Apply confirmation
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
                // fallback
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
