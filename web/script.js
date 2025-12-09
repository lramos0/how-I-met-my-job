/* --------------------------------------------------------
   PDF TEXT EXTRACTION
--------------------------------------------------------- */
async function extractPdfText(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    let finalText = "";

    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();

        let pageText = "";
        let lastY = null;

        for (const item of textContent.items) {
            const text = item.str;

            // detect line breaks using vertical movement
            if (lastY !== null && Math.abs(item.transform[5] - lastY) > 5) {
                pageText += "\n";
            }

            pageText += text;
            lastY = item.transform[5];
        }

        finalText += pageText + "\n\n";
    }

    // Clean up weird double-spacing PDF.js often leaves
    finalText = finalText
        .replace(/\s{2,}/g, " ")
        .replace(/\n{3,}/g, "\n\n")
        .trim();

    return finalText;
}

/* --------------------------------------------------------
   DOCX TEXT EXTRACTION
--------------------------------------------------------- */
async function extractDocxText(file) {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value;
}

/* --------------------------------------------------------
   FIX SMASHED WORDS
--------------------------------------------------------- */
function restoreWordBoundaries(text) {
    return text
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .replace(/([A-Z])([A-Z][a-z])/g, "$1 $2")
        .replace(/(\D)(\d)/g, "$1 $2")
        .replace(/(\d)(\D)/g, "$1 $2")
        .replace(/\s{2,}/g, " ")
        .trim();
}

/* --------------------------------------------------------
   SEND JSON → BACKEND
--------------------------------------------------------- */
async function sendJSON(bodyObj) {
    try {
        const response = await fetch("/.netlify/functions/classify-cv", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyObj)
        });

        const text = await response.text();
        try {
            return JSON.parse(text);
        } catch {
            console.error("Backend returned non-JSON:", text);
            return null;
        }
    } catch (e) {
        console.error("Backend error:", e);
        return null;
    }
}

let resumeJSON = null;

let skillDictPromise = null;
async function ensureSkillDict() {
    if (typeof SKILL_DICT === 'object') return SKILL_DICT;
    if (skillDictPromise) {
        await skillDictPromise;
        return typeof SKILL_DICT === 'object' ? SKILL_DICT : {};
    }

    skillDictPromise = new Promise(resolve => {
        // Avoid injecting multiple times
        const existing = document.querySelector('script[data-skill-dict="true"]');
        if (existing) {
            existing.addEventListener('load', () => resolve(typeof SKILL_DICT === 'object' ? SKILL_DICT : {}));
            existing.addEventListener('error', () => resolve({}));
            return;
        }

        const s = document.createElement('script');
        s.src = 'model/data/skill_dict.js';
        s.dataset.skillDict = 'true';
        s.onload = () => resolve(typeof SKILL_DICT === 'object' ? SKILL_DICT : {});
        s.onerror = () => resolve({});
        document.head.appendChild(s);
    });

    await skillDictPromise;
    return typeof SKILL_DICT === 'object' ? SKILL_DICT : {};
}

/* --------------------------------------------------------
   MAIN PARSER
--------------------------------------------------------- */
async function parseResume() {
    const fileInput = document.getElementById("resumeFile");
    const status = document.getElementById("statusMsg");

    if (!fileInput.files.length) {
        alert("Please upload a resume file first.");
        return;
    }

    const file = fileInput.files[0];
    status.textContent = `📄 Extracting text from ${file.name}…`;

    let resumeText = "";
    try {
        if (file.name.endsWith(".pdf")) {
            resumeText = await extractPdfText(file);
        } else if (file.name.endsWith(".docx")) {
            resumeText = await extractDocxText(file);
        } else {
            resumeText = await file.text();
        }
    } catch (err) {
        console.error(err);
        status.textContent = "❌ Error reading file.";
        return;
    }

    resumeText = restoreWordBoundaries(resumeText);
    console.log("Restored Resume Text:", resumeText);

    status.textContent = "🤖 Parsing resume…";

    const skills = await extractSkills(resumeText);

    resumeJSON = {
        inputs: [{
            candidate_id: "CAND-" + Date.now(),
            full_name: extractName(resumeText),
            location: extractLocation(resumeText),
            education_level: extractEducation(resumeText),
            years_experience: estimateYears(resumeText),
            skills,
            certifications: extractCerts(resumeText),
            current_title: extractTitle(resumeText),
            industries: extractIndustries(resumeText),
            achievements: extractAchievements(resumeText)
        }],
        password: "craig123"
    };

    status.textContent = "📡 Sending to backend…";

    const response = await sendJSON(resumeJSON);
    console.log("Backend response:", response);

    if (response && Array.isArray(response.predictions) && response.predictions.length > 0) {
        const prediction = response.predictions[0];
        resumeJSON.inputs[0].competitive_score = prediction.competitive_score;
    }

    // Now show the preview *after* you've set the score
    const previewSection = document.getElementById("previewSection");
    if (previewSection) {
        previewSection.classList.remove("d-none");
        // Also ensure it's visible if style.display was used
        previewSection.style.display = "block";
    }

    displayParsedResume();

    status.textContent = "✅ Resume parsed!";
}

/* --------------------------------------------------------
   DISPLAY PARSED RESUME
--------------------------------------------------------- */
function displayParsedResume() {
    const rec = resumeJSON.inputs[0];
    const outputDiv = document.getElementById("parsedOutput");
    if (!outputDiv) return;

    outputDiv.innerHTML = `
        <b>Location:</b> ${rec.location}<br>
        <b>Education:</b> ${rec.education_level}<br>
        <b>Years Experience:</b> ${rec.years_experience}<br>
        <b>Title:</b> ${rec.current_title}<br>
        <b>Skills:</b> ${rec.skills.join(", ") || "None detected"}<br>
        <b>Certifications:</b> ${rec.certifications.join(", ") || "None"}<br>
        <b>Industries:</b> ${rec.industries.join(", ") || "Unknown"}<br>
        <b>Achievements:</b> ${rec.achievements.join(", ") || "None"}<br>
        <b>Competitive Score:</b> ${rec.competitive_score !== undefined ? rec.competitive_score : "N/A"}
    `;
}

/* --------------------------------------------------------
   FIELD EXTRACTORS
--------------------------------------------------------- */
function extractName(text) {
    // Stronger: looks for "Firstname Lastname" OR "First M. Last"
    const match = text.match(/\b([A-Z][a-z]+)\s+(?:[A-Z]\.\s+)?([A-Z][a-z]+)\b/);
    return match ? `${match[1]} ${match[2]}` : "Unknown";
}

function extractLocation(text) {
    const match = text.match(/[A-Z][a-z]+,\s?[A-Z]{2}\b/);
    return match ? match[0] : "Unknown";
}

function extractEducation(text) {
    if (/master|m\.?s\.?\b/i.test(text)) return "Master";
    if (/bachelor|b\.?s\.?\b/i.test(text)) return "Bachelor";
    if (/ph\.?d|doctor/i.test(text)) return "PhD";
    return "Unknown";
}

async function extractSkills(text) {
    const dict = await ensureSkillDict();
    const lower = text.toLowerCase();
    const out = new Set();
    if (dict && typeof dict === 'object') {
        Object.keys(dict).forEach(canonical => {
            const aliases = dict[canonical] || [];
            for (let i = 0; i < aliases.length; i++) {
                const a = String(aliases[i]).toLowerCase();
                if (!a) continue;
                if (lower.indexOf(a) !== -1) {
                    out.add(canonical);
                    break;
                }
            }
        });
    }
    return Array.from(out);
}

function extractCerts(text) {
    const certs = [];
    if (/aws solutions architect/i.test(text)) certs.push("AWS Solutions Architect");
    if (/pmp/i.test(text)) certs.push("PMP");
    return certs;
}

function extractTitle(text) {
    const roles = [
        "Software Engineer", "ML Engineer", "Data Scientist", "Developer",
        "Teacher", "Counselor", "Supervisor", "Specialist", "Analyst",
        "Manager", "Director", "Administrator", "Coordinator", "Consultant"
    ];
    for (let r of roles) {
        const regex = new RegExp(r, "i");
        if (regex.test(text)) return r;
    }
    return "Unknown";
}

function estimateYears(text) {
    const matches = [...text.matchAll(/(\d+)\s+years?/gi)];
    return matches.length ? Math.max(...matches.map(m => parseInt(m[1]))) : 0;
}

function extractIndustries(text) {
    const industries = [];
    if (/education|teacher|child|school/i.test(text)) industries.push("Education");
    if (/software|developer/i.test(text)) industries.push("Software");
    if (/health|care/i.test(text)) industries.push("Healthcare");
    return industries;
}

function extractAchievements(text) {
    const achievements = [];
    if (/dean('|’)s list/i.test(text)) achievements.push("Dean's List");
    if (/chancellor('|’)s list/i.test(text)) achievements.push("Chancellor's List");
    return achievements;
}

function goToJobsPage() {
    localStorage.setItem("parsedResume", JSON.stringify(resumeJSON));
    window.location.href = "jobs.html";
}