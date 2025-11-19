// jobs.js — loads a CSV of job postings, renders cards, filters, chart, and shows loading/empty states

document.addEventListener('DOMContentLoaded', () => {
    window._jobsData = [];
    window._allDropdownData = {};
    initUI();
    loadDropdowns();
    
    // Add event listeners
    document.getElementById('sortBy')?.addEventListener('change', applySort);
    document.getElementById('globalSearch')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performGlobalSearch();
    });
});

function initUI() {
    // Insert spinner and results container if missing
    const results = document.getElementById('jobResults');
    if (!results) return;

    // add spinner
    let spinner = document.createElement('div');
    spinner.id = 'loadingSpinner';
    spinner.className = 'd-none text-center my-4';
    spinner.innerHTML = '<div class="spinner-border" role="status" aria-hidden="true"></div>';
    results.parentElement.insertBefore(spinner, results);

    // render initial empty state
    results.innerHTML = '';
}

async function loadDropdowns() {
    showSpinner(true);
    try {
        const res = await fetch('/api/dropdowns');
        if (!res.ok) throw new Error('Failed to load dropdown options');
        const dropdowns = await res.json();
        window._allDropdownData = dropdowns;
        populateDropdownsFromAPI(dropdowns);
        loadInitialJobs();
    } catch (err) {
        showError(err.message);
    } finally {
        showSpinner(false);
    }
}

async function loadInitialJobs() {
    showSpinner(true);
    try {
        const res = await fetch('/api/jobs?limit=50');
        if (!res.ok) throw new Error('Failed to load jobs');
        const jobs = await res.json();
        window._jobsData = jobs;
        renderJobs(jobs);
        renderChart(jobs);
    } catch (err) {
        showError(err.message);
    } finally {
        showSpinner(false);
    }
}

function showSpinner(show) {
    const s = document.getElementById('loadingSpinner');
    if (!s) return;
    s.classList.toggle('d-none', !show);
}

function showError(msg) {
    const results = document.getElementById('jobResults');
    results.innerHTML = `<div class="alert alert-danger">${escapeHtml(msg)}</div>`;
}

function populateDropdownsFromAPI(data) {
    // data = { companies: [...], titles: [...], locations: [...], industries: [...] }
    const companies = data.companies || [];
    const titles = data.titles || [];
    const locations = data.locations || [];
    const industries = data.industries || [];

    console.log('Companies:', companies.length, companies.slice(0, 3));
    console.log('Titles:', titles.length, titles.slice(0, 3));
    console.log('Locations:', locations.length, locations.slice(0, 3));
    console.log('Industries:', industries.length, industries.slice(0, 3));

    // Populate dropdowns
    const companySelect = document.getElementById('filterCompany');
    companies.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        companySelect.appendChild(opt);
    });

    const titleSelect = document.getElementById('filterTitle');
    titles.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        titleSelect.appendChild(opt);
    });

    const locationSelect = document.getElementById('filterLocation');
    locations.forEach(l => {
        const opt = document.createElement('option');
        opt.value = l;
        opt.textContent = l;
        locationSelect.appendChild(opt);
    });

    const industrySelect = document.getElementById('filterIndustry');
    industries.forEach(i => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = i;
        industrySelect.appendChild(opt);
    });
}

function clearFilters() {
    ['filterCompany','filterTitle','filterLocation','filterIndustry'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    document.getElementById('minSalary').value = '';
    document.getElementById('maxSalary').value = '';
    document.getElementById('globalSearch').value = '';
    document.getElementById('sortBy').value = 'relevance';
    loadInitialJobs();
}

function performGlobalSearch() {
    const search = (document.getElementById('globalSearch')?.value || '').trim();
    if (!search) {
        showError('Please enter a search term');
        return;
    }

    showSpinner(true);
    
    const params = new URLSearchParams();
    params.append('search', search);
    params.append('limit', 100);

    fetch(`/api/jobs?${params.toString()}`)
        .then(res => res.json())
        .then(filtered => {
            console.log('Search results:', filtered.length);
            window._jobsData = filtered;
            applySort();
            showSpinner(false);
        })
        .catch(err => {
            showError(err.message);
            showSpinner(false);
        });
}

function applySort() {
    const sortBy = document.getElementById('sortBy')?.value || 'relevance';
    const data = window._jobsData || [];

    let sorted = [...data];

    switch(sortBy) {
        case 'score_desc':
            sorted.sort((a, b) => computeMatchScore(b) - computeMatchScore(a));
            break;
        case 'score_asc':
            sorted.sort((a, b) => computeMatchScore(a) - computeMatchScore(b));
            break;
        case 'company':
            sorted.sort((a, b) => (a.company_name || '').localeCompare(b.company_name || ''));
            break;
        case 'date_new':
            sorted.sort((a, b) => new Date(b.job_posted_date || 0) - new Date(a.job_posted_date || 0));
            break;
        case 'salary_high':
            sorted.sort((a, b) => {
                const aSal = extractSalary(a.job_base_pay_range || '');
                const bSal = extractSalary(b.job_base_pay_range || '');
                return bSal - aSal;
            });
            break;
        case 'relevance':
        default:
            sorted.sort((a, b) => computeMatchScore(b) - computeMatchScore(a));
    }

    renderJobs(sorted);
    renderChart(sorted);
}

function extractSalary(salaryStr) {
    try {
        const numbers = salaryStr.match(/[\d,]+\.?\d*/g);
        if (numbers && numbers.length > 0) {
            return parseFloat(numbers[0].replace(/,/g, ''));
        }
    } catch (e) {}
    return 0;
}

function filterJobs() {
    const company = (document.getElementById('filterCompany')?.value || '').trim();
    const title = (document.getElementById('filterTitle')?.value || '').trim();
    const location = (document.getElementById('filterLocation')?.value || '').trim();
    const industry = (document.getElementById('filterIndustry')?.value || '').trim();
    const minSalary = parseInt(document.getElementById('minSalary')?.value || '0') || 0;
    const maxSalary = parseInt(document.getElementById('maxSalary')?.value || '999999999') || 999999999;

    showSpinner(true);
    
    // Build query string
    const params = new URLSearchParams();
    if (company) params.append('company', company);
    if (title) params.append('title', title);
    if (location) params.append('location', location);
    if (industry) params.append('industry', industry);
    if (minSalary > 0) params.append('minSalary', minSalary);
    if (maxSalary < 999999999) params.append('maxSalary', maxSalary);
    params.append('limit', 100);

    fetch(`/api/jobs?${params.toString()}`)
        .then(res => res.json())
        .then(filtered => {
            console.log('Filtered results:', filtered.length);
            window._jobsData = filtered;
            applySort();
            showSpinner(false);
        })
        .catch(err => {
            showError(err.message);
            showSpinner(false);
        });
}

function renderJobs(data) {
    const results = document.getElementById('jobResults');
    if (!results) return;
    if (!data || data.length === 0) {
        results.innerHTML = '<div class="text-center py-4">No matching jobs found.</div>';
        return;
    }

    // Render as Bootstrap cards
    const cards = data.map((row, idx) => {
        const score = computeMatchScore(row);
        return `
        <div class="card mb-3">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5 class="card-title mb-1">${escapeHtml(row.job_title || 'Untitled')}</h5>
                        <h6 class="card-subtitle mb-2 text-muted">${escapeHtml(row.company_name || '')} — ${escapeHtml(row.job_location || '')}</h6>
                        <p class="card-text small mb-1">${escapeHtml((row.job_summary || '').substring(0, 200))}...</p>
                        <p class="card-text small"><strong>Industry:</strong> ${escapeHtml(row.job_industries || '—')}</p>
                    </div>
                    <div class="text-end">
                        <div class="fs-3 fw-bold">${score}%</div>
                        <div class="small text-muted">Match Score</div>
                    </div>
                </div>
            </div>
        </div>
        `;
    }).join('\n');

    results.innerHTML = cards;
}

function computeMatchScore(row) {
    // Lightweight heuristic demo: match based on presence of common keywords
    const keywords = ['python','sql','aws','react','java','javascript','excel','machine learning','nlp','tensorflow','pandas'];
    const text = ((row.job_title||'') + ' ' + (row.job_summary||'') + ' ' + (row.job_industries||'')).toLowerCase();
    let matches = 0;
    keywords.forEach(k => { if (text.includes(k)) matches++; });
    const score = Math.min(100, Math.round((matches / keywords.length) * 100) + 50); // base 50
    return score;
}

function renderChart(data) {
    const ctx = document.getElementById('matchChart').getContext('2d');
    const top = (data || []).slice(0, 10).map(r => ({
        label: r.title || (r.company || 'Job'),
        score: computeMatchScore(r)
    }));

    // Destroy existing chart if present
    if (window._matchChart) {
        window._matchChart.destroy();
    }

    window._matchChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: top.map(t => t.label),
            datasets: [{
                label: 'Match Score',
                data: top.map(t => t.score),
                backgroundColor: 'rgba(54, 162, 235, 0.6)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 100 } }
        }
    });
}

function escapeHtml(s) {
    if (!s) return '';
    return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
