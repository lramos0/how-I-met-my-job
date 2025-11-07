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

    const jobs = await fetch("data/job_postings.csv").then(res => res.text());
    window.jobData = csvToArray(jobs);
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

function filterJobs() {
    const company = document.getElementById("filterCompany").value.toLowerCase();
    const title = document.getElementById("filterTitle").value.toLowerCase();
    const location = document.getElementById("filterLocation").value.toLowerCase();
    const industry = document.getElementById("filterIndustry").value.toLowerCase();

    let filtered = window.jobData.filter(job =>
        (!company || job.company_name?.toLowerCase().includes(company)) &&
        (!title || job.job_title?.toLowerCase().includes(title)) &&
        (!location || job.job_location?.toLowerCase().includes(location)) &&
        (!industry || job.job_industries?.toLowerCase().includes(industry))
    );

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

loadResume();
