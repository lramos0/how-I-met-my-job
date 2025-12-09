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
    const original = text;
    text = text.toLowerCase();

    const contains = (regex) => regex.test(text);

    // ============================================================
    //                       PH.D / DOCTORAL
    // ============================================================
    const phdRegex = [
        /\bph\.?\s*d\.?\b/,                                   // PhD / Ph.D.
        /\bdoctorate\b/,
        /\bdoctor of\b/,
        /\bdoctors in\b/,
        /\bdphil\b/,
        /\bsc\.?\s*d\.?\b/,                                   // ScD

        // Law / JD
        /\bj\.?\s*d\.?\b/,
        /\bjuris doctor\b/,
        /\blaw school\b/,

        // Medical / Clinical doctor-level
        /\bmd\b|\bm\.?\s*d\.?\b/,                             // MD
        /\bdo\b(?!\b.*not)/,                                  // DO (doctor of osteopathy)
        /\bd\.?\s*d\.?\s*s\.?\b/,                             // DDS
        /\bdmd\b/,                                            // DMD dentistry
        /\bpharmd\b/,                                         // PharmD
        /\bdnp\b/,                                            // Doctor of Nursing Practice
        /\bpsy\.?\s*d\.?\b|\bpsychology doctorate\b/,
        /\bjunior doctor\b/
    ];

    if (phdRegex.some(r => r.test(text))) {
        return "Doctorate";
    }

    // ============================================================
    //                          MASTER
    // ============================================================
    const masterPositive = [
        /\bmaster['’]s\b/,
        /\bmaster of\b/,

        /\bms\b|\bm\.?\s*s\.?\b/,
        /\bma\b|\bm\.?\s*a\.?\b/,
        /\bmsc\b/,
        /\bmba\b/,
        /\bmeng\b|\bm\.?\s*eng\.?\b/,
        /\bm\.?ed\.?\b/,

        // Nursing
        /\bmsn\b/,                    // Masters in Nursing
        /\bnurse practitioner\b/,     // NP usually masters-level
    ];

    const masterNegative = [
        /\b(master bedroom|master plan|master key)\b/,
        /\b(headmaster|grandmaster|master electrician)\b/
    ];

    if (
        masterPositive.some(r => r.test(text)) &&
        !masterNegative.some(r => r.test(text))
    ) {
        return "Master";
    }

    // ============================================================
    //                          BACHELOR
    // ============================================================
    const bachelorPositive = [
        /\bbachelor['’]s\b/,
        /\bbachelor of\b/,

        /\bbs\b|\bb\.?\s*s\.?\b/,
        /\bba\b|\bb\.?\s*a\.?\b/,
        /\bbsc\b/,
        /\bbfa\b/,
        /\bbeng\b/,

        /\bbe\b|\bb\.?\s*e\.?\b/,

        // Nursing
        /\bbsn\b/, // Bachelor of Science in Nursing
    ];

    const bachelorNegative = [
        /\bbachelor party\b/,
        /\bbachelor pad\b/
    ];

    if (
        bachelorPositive.some(r => r.test(text)) &&
        !bachelorNegative.some(r.test(text))
    ) {
        return "Bachelor";
    }

    // ============================================================
    //                    ASSOCIATES DEGREE
    // ============================================================
    const associateRegex = [
        /\bassociate['’]s\b/,
        /\bassociate of\b/,
        /\baa\b\b|\ba\.?\s*a\.?\b/,
        /\bas\b\b|\ba\.?\s*s\.?\b/,
        /\baas\b/,      // Associate of Applied Science
        /\baos\b/,      // Associate of Occupational Studies
    ];

    if (associateRegex.some(r => r.test(text))) {
        return "Associate";
    }

    // ============================================================
    //                      NURSING PROGRAMS
    // ============================================================
    const nursingRegex = [
        /\brn\b(?!\w)/,                 // RN
        /\blicensed practical nurse\b/,
        /\blicensed vocational nurse\b/,
        /\blpn\b|\blvn\b/,
        /\bcna\b/,                      // Certified Nursing Assistant
        /\bpnp\b/,                      // Pediatric NP (if not classified above)
        /\baprn\b/,                     // advanced practice nurse
    ];

    if (nursingRegex.some(r => r.test(text))) {
        return "Nursing Program";
    }

    // ============================================================
    //                          TRADE SCHOOL
    // ============================================================
    const tradeSchoolRegex = [
        /\btrade school\b/,
        /\btechnical school\b/,
        /\btech school\b/,
        /\bcommunity college certificate\b/,

        // Trade occupations
        /\bhvac\b/,
        /\bwelding\b|\bweld(er|ing)\b/,
        /\belectrician\b/,
        /\bplumbing\b|\bplumber\b/,
        /\bcarpentry\b|\bcarpenter\b/,
        /\bautomotive\b/,
        /\bmechanic training\b/,
        /\bcosmetology\b/,
        /\bculinary school\b/,
    ];

    if (tradeSchoolRegex.some(r => r.test(text))) {
        return "Trade School / Vocational";
    }

    // ============================================================
    //                  MILITARY EDUCATION / TRAINING
    // ============================================================
    const militaryRegex = [
        /\bmilitary training\b/,
        /\barmy education\b/,
        /\bnavy school\b/,
        /\bair force academy\b/,
        /\bmarine corps school\b/,
        /\bcoast guard school\b/,
        /\bboot camp\b/,
        /\bmos school\b/,             // Army MOS training
        /\brate school\b/,            // Navy rate training
        /\bmilitary occupational specialty\b/,
    ];

    if (militaryRegex.some(r => r.test(text))) {
        return "Military Education / Training";
    }

    // ============================================================
    //                  CERTIFICATIONS / TRAINING PROGRAMS
    // ============================================================
    const certificateRegex = [
        /\bcertificate\b/,
        /\bcertification\b/,
        /\bcertified\b/,
        /\btraining program\b/,
        /\bprofessional training\b/,
        /\bcontinuing education\b/,
        /\bbootcamp\b/,
        /\bworkshop\b/,
    ];

    if (certificateRegex.some(r => r.test(text))) {
        return "Certificate / Training Program";
    }

    // ============================================================
    //                         FALLBACK
    // ============================================================
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


const monthIndex = {
  jan: 0, january: 0,
  feb: 1, february: 1,
  mar: 2, march: 2,
  apr: 3, april: 3,
  may: 4,
  jun: 5, june: 5,
  jul: 6, july: 6,
  aug: 7, august: 7,
  sep: 8, sept: 8, september: 8,
  oct: 9, october: 9,
  nov: 10, november: 10,
  dec: 11, december: 11
};

function parseDateToken(token, isEnd) {
  token = token.trim();

  // Present / Current / Now
  if (/^(present|current|now)$/i.test(token)) return new Date();

  // Month Year
  let m = token.match(
    /(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ ,.-]*(\d{4})/i
  );
  if (m) {
    const month = monthIndex[m[1].toLowerCase()];
    const year = parseInt(m[2], 10);
    return new Date(year, month, 1);
  }

  // MM/YYYY
  m = token.match(/(\d{1,2})[/-](\d{4})/);
  if (m) {
    const month = Math.min(Math.max(parseInt(m[1], 10) - 1, 0), 11);
    const year = parseInt(m[2], 10);
    return new Date(year, month, 1);
  }

  // Bare year "2018"
  m = token.match(/(\d{4})/);
  if (m) {
    const year = parseInt(m[1], 10);
    return new Date(year, isEnd ? 11 : 0, 1);
  }

  return null;
}

function estimateYears(text) {
  const explicitMatches = [...text.matchAll(/(\d+(?:\.\d+)?)\s+years?/gi)];
  const explicitYears = explicitMatches.length
    ? Math.max(...explicitMatches.map(m => parseFloat(m[1])))
    : 0;

  const RANGE_REGEX =
    /((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ ,.-]*\d{4}|\d{1,2}[/-]\d{4}|\d{4})\s*(?:-|–|to)\s*(Present|Current|Now|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ ,.-]*\d{4}|\d{1,2}[/-]\d{4}|\d{4})/gi;

  const ranges = [];

  for (const match of text.matchAll(RANGE_REGEX)) {
    const start = parseDateToken(match[1], false);
    const end = parseDateToken(match[2], true);
    if (start && end && end >= start) {
      ranges.push({ start, end });
    }
  }

  if (!ranges.length) return Math.floor(explicitYears);

  // Merge overlapping ranges
  ranges.sort((a, b) => a.start - b.start);
  const merged = [];
  let cur = ranges[0];

  for (let i = 1; i < ranges.length; i++) {
    const next = ranges[i];
    if (next.start <= cur.end) {
      if (next.end > cur.end) cur.end = next.end;
    } else {
      merged.push(cur);
      cur = next;
    }
  }
  merged.push(cur);

  // Compute months
  let totalMonths = 0;
  for (const r of merged) {
    const months =
      (r.end.getFullYear() - r.start.getFullYear()) * 12 +
      (r.end.getMonth() - r.start.getMonth()) +
      1;
    totalMonths += Math.max(months, 0);
  }

  const yearsFromDates = totalMonths / 12;
  return Math.floor(Math.max(yearsFromDates, explicitYears));
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