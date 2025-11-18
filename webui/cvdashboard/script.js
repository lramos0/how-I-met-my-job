// async function parseResume() {
//     const fileInput = document.getElementById('resumeFile');
//     const file = fileInput.files[0];
//     if (!file) return alert("Please upload a file first!");

//     let textContent = "";

//     if (file.name.endsWith('.pdf')) {
//         const arrayBuffer = await file.arrayBuffer();
//         const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

//         for (let i = 1; i <= pdf.numPages; i++) {
//             const page = await pdf.getPage(i);
//             const text = await page.getTextContent();
//             textContent += text.items.map(s => s.str).join(' ') + '\n';
//         }
//     } else if (file.name.endsWith('.docx')) {
//         const arrayBuffer = await file.arrayBuffer();
//         const result = await mammoth.extractRawText({ arrayBuffer });
//         textContent = result.value;
//     }

//     document.getElementById('previewSection').style.display = 'block';
//     document.getElementById('resumeText').textContent = textContent;

//     const jsonData = extractFields(textContent);
//     window.resumeJSON = jsonData;

//     document.getElementById('jsonOutput').textContent = JSON.stringify(jsonData, null, 2);
// }

// function extractFields(text) {
//     const skillsRegex = /(skills?|technologies?):([\s\S]*?)(education|experience|projects|$)/i;
//     const eduRegex = /(education|qualifications):([\s\S]*?)(experience|projects|skills|$)/i;
//     const expRegex = /(experience|employment|work history):([\s\S]*?)(education|skills|projects|$)/i;

//     const skillsSection = skillsRegex.test(text) ? skillsRegex.exec(text)[2].trim() : "";
//     const skillsList = skillsSection.split(/,|;|\n/).map(s => s.trim().toLowerCase()).filter(Boolean);

//     return {
//         name: text.match(/([A-Z][a-z]+\s[A-Z][a-z]+)/)?.[0] || "Unknown",
//         email: text.match(/[a-zA-Z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/)?.[0] || "Not found",
//         skills: skillsList,
//         education: eduRegex.test(text) ? eduRegex.exec(text)[2].trim() : "Not found",
//         experience: expRegex.test(text) ? expRegex.exec(text)[2].trim() : "Not found",
//     };
// }

// function calculateMatch() {
//     const query = document.getElementById('searchBox').value.toLowerCase().split(',').map(s => s.trim()).filter(Boolean);
//     if (!query.length) return alert("Enter skills to compare!");

//     const resumeSkills = window.resumeJSON?.skills || [];
//     const matched = query.filter(skill => resumeSkills.includes(skill));
//     const score = ((matched.length / query.length) * 100).toFixed(1);

//     // Update Progress Bar
//     const bar = document.getElementById('matchProgress');
//     bar.style.width = `${score}%`;
//     bar.textContent = `${score}%`;

//     const details = document.getElementById('matchDetails');
//     details.textContent = `Matched Skills: ${matched.join(', ') || 'None'}`;

//     document.getElementById('matchContainer').style.display = 'block';

//     // Create Skills Chart
//     const ctx = document.getElementById('skillsChart').getContext('2d');
//     if (window.skillChart) window.skillChart.destroy();
//     window.skillChart = new Chart(ctx, {
//         type: 'doughnut',
//         data: {
//             labels: ['Matched', 'Unmatched'],
//             datasets: [{
//                 data: [matched.length, query.length - matched.length],
//                 backgroundColor: ['#4caf50', '#f44336']
//             }]
//         },
//         options: { plugins: { legend: { position: 'bottom' } } }
//     });
// }

// function downloadJSON() {
//     const blob = new Blob([JSON.stringify(window.resumeJSON, null, 2)], { type: "application/json" });
//     saveAs(blob, "parsed_resume.json");
// }
// function goToJobsPage() {
//     localStorage.setItem("parsedResume", JSON.stringify(window.resumeJSON));
//     window.location.href = "jobs.html";
// }

function calculateMatch() {
    // keep this if you use it on jobs.html
    const query = document.getElementById('searchBox').value.toLowerCase().split(',').map(s => s.trim()).filter(Boolean);
    if (!query.length) return alert("Enter skills to compare!");
  
    const resumeData = JSON.parse(localStorage.getItem("parsedResume") || "{}");
    const resumeSkills = resumeData.skills || [];
    const matched = query.filter(skill => resumeSkills.includes(skill));
    const score = ((matched.length / query.length) * 100).toFixed(1);
  
    const bar = document.getElementById('matchProgress');
    bar.style.width = `${score}%`;
    bar.textContent = `${score}%`;
  
    const details = document.getElementById('matchDetails');
    details.textContent = `Matched Skills: ${matched.join(', ') || 'None'}`;
  }
  