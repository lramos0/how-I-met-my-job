async function parseResume() {
    const fileInput = document.getElementById('resumeFile');
    const file = fileInput.files[0];
    if (!file) return alert("Please upload a file first!");

    // Prefer a configured Cloud Function URL first, then local server, then client-side parsing
    const cloudUrl = window.CLOUD_FUNCTION_URL || localStorage.getItem('cloudFunctionUrl') || localStorage.getItem('CLOUD_FUNCTION_URL');
    if (cloudUrl) {
        try {
            const cloudResult = await sendToCloudFunction(cloudUrl, file);
            if (cloudResult && cloudResult.parsed) {
                window.resumeJSON = cloudResult.parsed;
                document.getElementById('previewSection').style.display = 'block';
                document.getElementById('parsedOutput').innerHTML = `\n      <b>Name:</b> ${window.resumeJSON.name}<br>\n      <b>Email:</b> ${window.resumeJSON.email}<br>\n      <b>Skills:</b> ${(window.resumeJSON.skills || []).join(', ')}<br>\n      <b>Education:</b> ${window.resumeJSON.education || 'N/A'}<br>\n      <b>Experience:</b> ${window.resumeJSON.experience || 'N/A'}\n    `;
                localStorage.setItem('parsedResume', JSON.stringify(window.resumeJSON));
                return;
            }
        } catch (e) {
            console.warn('Cloud function parse failed, will try local API then client fallback', e);
        }
    }

    // Try local server parse next
    try {
        const serverResult = await sendToLocalParseApi('http://127.0.0.1:5001/parse-resume', file);
        if (serverResult && serverResult.parsed) {
            window.resumeJSON = serverResult.parsed;
            document.getElementById('previewSection').style.display = 'block';
            document.getElementById('parsedOutput').innerHTML = `\n      <b>Name:</b> ${window.resumeJSON.name}<br>\n      <b>Email:</b> ${window.resumeJSON.email}<br>\n      <b>Skills:</b> ${(window.resumeJSON.skills || []).join(', ')}<br>\n      <b>Education:</b> ${window.resumeJSON.education || 'N/A'}<br>\n      <b>Experience:</b> ${window.resumeJSON.experience || 'N/A'}\n    `;
            localStorage.setItem('parsedResume', JSON.stringify(window.resumeJSON));
            return;
        }
    } catch (e) {
        console.warn('Local parse API not available, falling back to client parsing');
    }

    // Fallback: client-side parsing (pdf.js + mammoth)
    let textContent = await getTextFromFile(file);
    document.getElementById('previewSection').style.display = 'block';
    // if there's an element 'resumeText' use it, otherwise update parsedOutput
    const resumeTextEl = document.getElementById('resumeText');
    if (resumeTextEl) resumeTextEl.textContent = textContent;

    const jsonData = extractFields(textContent);
    window.resumeJSON = jsonData;

    const jsonOut = document.getElementById('jsonOutput');
    if (jsonOut) jsonOut.textContent = JSON.stringify(jsonData, null, 2);
    const out = document.getElementById('parsedOutput');
    if (out) out.innerHTML = `\n      <b>Name:</b> ${jsonData.name}<br>\n      <b>Email:</b> ${jsonData.email}<br>\n      <b>Skills:</b> ${(jsonData.skills || []).join(', ')}<br>\n      <b>Education:</b> ${jsonData.education || 'N/A'}<br>\n      <b>Experience:</b> ${jsonData.experience || 'N/A'}\n    `;
}

async function getTextFromFile(file) {
    let textContent = "";
    if (file.name.endsWith('.pdf')) {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const text = await page.getTextContent();
            textContent += text.items.map(s => s.str).join(' ') + '\n';
        }
    } else if (file.name.endsWith('.docx')) {
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        textContent = result.value;
    }
    return textContent;
}

function extractFields(text) {
    const skillsRegex = /(skills?|technologies?):([\s\S]*?)(education|experience|projects|$)/i;
    const eduRegex = /(education|qualifications):([\s\S]*?)(experience|projects|skills|$)/i;
    const expRegex = /(experience|employment|work history):([\s\S]*?)(education|skills|projects|$)/i;

    const skillsSection = skillsRegex.test(text) ? skillsRegex.exec(text)[2].trim() : "";
    const skillsList = skillsSection.split(/,|;|\n/).map(s => s.trim().toLowerCase()).filter(Boolean);

    return {
        name: text.match(/([A-Z][a-z]+\s[A-Z][a-z]+)/)?.[0] || "Unknown",
        email: text.match(/[a-zA-Z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/)?.[0] || "Not found",
        skills: skillsList,
        education: eduRegex.test(text) ? eduRegex.exec(text)[2].trim() : "Not found",
        experience: expRegex.test(text) ? expRegex.exec(text)[2].trim() : "Not found",
    };
}

function calculateMatch() {
    const query = document.getElementById('searchBox').value.toLowerCase().split(',').map(s => s.trim()).filter(Boolean);
    if (!query.length) return alert("Enter skills to compare!");

    const resumeSkills = window.resumeJSON?.skills || [];
    const matched = query.filter(skill => resumeSkills.includes(skill));
    const score = ((matched.length / query.length) * 100).toFixed(1);

    // Update Progress Bar
    const bar = document.getElementById('matchProgress');
    bar.style.width = `${score}%`;
    bar.textContent = `${score}%`;

    const details = document.getElementById('matchDetails');
    details.textContent = `Matched Skills: ${matched.join(', ') || 'None'}`;

    document.getElementById('matchContainer').style.display = 'block';

    // Create Skills Chart
    const ctx = document.getElementById('skillsChart').getContext('2d');
    if (window.skillChart) window.skillChart.destroy();
    window.skillChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Matched', 'Unmatched'],
            datasets: [{
                data: [matched.length, query.length - matched.length],
                backgroundColor: ['#4caf50', '#f44336']
            }]
        },
        options: { plugins: { legend: { position: 'bottom' } } }
    });
}

function downloadJSON() {
    const blob = new Blob([JSON.stringify(window.resumeJSON, null, 2)], { type: "application/json" });
    saveAs(blob, "parsed_resume.json");
}
function goToJobsPage() {
    localStorage.setItem("parsedResume", JSON.stringify(window.resumeJSON));
    window.location.href = "jobs.html";
}

// POST file to local Flask parse API and return { parsed }
async function sendToLocalParseApi(url, file, timeoutMs = 4000) {
    try {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeoutMs);
        const form = new FormData();
        form.append('file', file, file.name);

        const resp = await fetch(url, { method: 'POST', body: form, signal: controller.signal });
        clearTimeout(id);
        if (!resp.ok) {
            // server may return 404 for root path or other errors
            return null;
        }
        const data = await resp.json();
        return { parsed: data };
    } catch (e) {
        return null;
    }
}

// POST file to Cloud Function URL and return { parsed }
async function sendToCloudFunction(url, file, timeoutMs = 10000) {
    try {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeoutMs);
        const form = new FormData();
        form.append('file', file, file.name);

        const resp = await fetch(url, { method: 'POST', body: form, signal: controller.signal });
        clearTimeout(id);
        if (!resp.ok) {
            return null;
        }
        const data = await resp.json();
        return { parsed: data };
    } catch (e) {
        return null;
    }
}