/* --------------------------------------------------------
   DYNAMIC STATUS (BACKEND WAITING)
--------------------------------------------------------- */
const backendStatusMessages = [
    "📡 Sending to backend…",
    "📖 Reading your resume…",
    "🏷️ Determining your industry…",
    "🛠️ Parsing your skills…",
    "📊 Crunching some numbers…",
    "☁️ Warming up compute clusters…",
    "🤝 Matching you with roles…"
];

let backendStatusTimer = null;

/**
 * Starts cycling through status messages every 2.5 seconds.
 * @param {HTMLElement} statusEl - The HTML element showing the status text.
 */
function startBackendStatusAnimation(statusEl) {
    if (!statusEl) return;

    let i = 0;
    statusEl.textContent = backendStatusMessages[i];

    // Clear any previous timer
    if (backendStatusTimer) {
        clearInterval(backendStatusTimer);
    }

    backendStatusTimer = setInterval(() => {
        i = (i + 1) % backendStatusMessages.length;
        statusEl.textContent = backendStatusMessages[i];
    }, 2500); // Change delay as desired
}

/**
 * Stops the rotating status messages and optionally sets a final text.
 * @param {HTMLElement} statusEl - The status element.
 * @param {string} finalText - (Optional) final message to display.
 */
function stopBackendStatusAnimation(statusEl, finalText) {
    if (backendStatusTimer) {
        clearInterval(backendStatusTimer);
        backendStatusTimer = null;
    }

    if (statusEl && finalText) {
        statusEl.textContent = finalText;
    }
}


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

    startBackendStatusAnimation(status);

    const response = await sendJSON(resumeJSON);
    console.log("Backend response:", response);

    stopBackendStatusAnimation(status);


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
function extractName() {
    const firstNames = FIRST_NAME_PARTS;
    const lastNames = LAST_NAME_PARTS;
    const first = firstNames[Math.floor(Math.random() * firstNames.length)];
    const last = lastNames[Math.floor(Math.random() * lastNames.length)];
    return `${first} ${last}`;
}

const FIRST_NAME_PARTS = [
    "Aaron", "Abigail", "Adam", "Adrian", "Aiden", "Alan", "Albert", "Alex", "Alexander", "Alexis",
    "Amanda", "Amber", "Amy", "Andrea", "Andrew", "Angela", "Anna", "Anthony", "Ashley", "Austin",
    "Benjamin", "Beth", "Blake", "Brandon", "Brenda", "Brian", "Brittany", "Bruce", "Bryan", "Caleb",
    "Cameron", "Carl", "Carol", "Carolyn", "Catherine", "Charles", "Charlotte", "Chelsea", "Chloe", "Chris",
    "Christian", "Christina", "Christopher", "Cindy", "Clarence", "Claire", "Clifford", "Cody", "Connor", "Courtney",
    "Crystal", "Cynthia", "Dakota", "Dale", "Dan", "Daniel", "Danielle", "David", "Deborah", "Dennis",
    "Diana", "Diane", "Donald", "Donna", "Doris", "Dorothy", "Douglas", "Dylan", "Edward", "Elijah",
    "Elizabeth", "Ellen", "Emily", "Emma", "Eric", "Ethan", "Eugene", "Evelyn", "Frank", "Gabriel",
    "Gary", "Gerald", "Gloria", "Grace", "Gregory", "Hannah", "Harold", "Heather", "Helen", "Henry",
    "Jack", "Jacob", "Jade", "James", "Jane", "Janet", "Janice", "Jason", "Jayden", "Jean",
    "Jeffrey", "Jennifer", "Jeremy", "Jerry", "Jesse", "Jessica", "Joan", "Joe", "John", "Johnny",
    "Jonathan", "Jordan", "Joseph", "Joshua", "Joyce", "Juan", "Judith", "Judy", "Julia", "Julie",
    "Justin", "Katherine", "Kathleen", "Kathryn", "Kayla", "Keith", "Kelly", "Kenneth", "Kevin", "Kimberly",
    "Kyle", "Larry", "Laura", "Lauren", "Lawrence", "Linda", "Lisa", "Logan", "Lori", "Louis",
    "Madison", "Margaret", "Maria", "Marie", "Marilyn", "Mark", "Martha", "Mary", "Matthew", "Megan",
    "Melissa", "Michael", "Michelle", "Mildred", "Morgan", "Nancy", "Nathan", "Nicholas", "Nicole", "Noah",
    "Olivia", "Pamela", "Patricia", "Patrick", "Paul", "Paula", "Peter", "Philip", "Rachel", "Ralph",
    "Randy", "Raymond", "Rebecca", "Richard", "Riley", "Robert", "Roger", "Ronald", "Rose", "Roy",
    "Russell", "Ruth", "Ryan", "Samantha", "Samuel", "Sandra", "Sara", "Sarah", "Scott", "Sean",
    "Sharon", "Shawn", "Shirley", "Sophia", "Stephanie", "Stephen", "Steven", "Susan", "Tammy", "Taylor",
    "Teresa", "Terry", "Theresa", "Thomas", "Timothy", "Tyler", "Victoria", "Vincent", "Virginia", "Walter",
    "Wayne", "William", "Zachary", "Aaliyah", "Abel", "Abram", "Ada", "Addison", "Adeline", "Ahmad",
    "Alana", "Alec", "Alyssa", "Amelia", "Amir", "Anastasia", "Andre", "Angel", "Angelo", "Anita",
    "Antonio", "Aria", "Ariana", "Arthur", "Athena", "Audrey", "Ava", "Avery", "Barry", "Beatrice",
    "Belinda", "Ben", "Bernard", "Bernice", "Bethany", "Beverly", "Bill", "Billy", "Blanche", "Bob",
    "Bradley", "Brady", "Brandy", "Brent", "Brett", "Brianna", "Bridget", "Brittney", "Brooke", "Bryce",
    "Byron", "Caitlin", "Carla", "Carlos", "Carmen", "Carrie", "Casey", "Cassandra", "Cecil", "Cecilia",
    "Cesar", "Chad", "Charlene", "Charlie", "Chester", "Cheyenne", "Claudia", "Clayton", "Colin", "Colleen",
    "Corey", "Craig", "Daisy", "Dallas", "Damian", "Damon", "Dana", "Danny", "Darlene", "Darrell",
    "Darren", "Daryl", "Dave", "Deanna", "Debra", "Devin", "Dewayne", "Diana", "Dolores", "Dominic",
    "Don", "Drew", "Duane", "Dustin", "Dwayne", "Earl", "Eduardo", "Eileen", "Elaine", "Eli",
    "Ella", "Elliott", "Erica", "Erick", "Erika", "Erin", "Ernest", "Esther", "Eva", "Evan",
    "Faith", "Felicia", "Fernando", "Florence", "Francis", "Fred", "Frederick", "Gail", "Gavin", "Gayle",
    "Gene", "Georgia", "Gilbert", "Gina", "Glen", "Glenn", "Gordon", "Grant", "Grayson", "Guy",
    "Gwendolyn", "Hailey", "Haley", "Hazel", "Hector", "Holly", "Hope", "Howard", "Hunter", "Ian",
    "Irene", "Iris", "Isaac", "Isabel", "Ivan", "Jacqueline", "Jade", "Jake", "Jared", "Jasmine",
    "Javier", "Jay", "Jeanette", "Jeff", "Jenna", "Jennie", "Jeremiah", "Jill", "Jimmie", "Joann",
    "Jodi", "Jody", "Jon", "Jordan", "Jorge", "Josephine", "Joy", "Juanita", "Julian", "June",
    "Kaitlyn", "Kara", "Karen", "Karl", "Kate", "Katelyn", "Katie", "Kaylee", "Keisha", "Kelsey",
    "Kendra", "Kenny", "Kent", "Kerry", "Kirk", "Kurt", "Kylie", "Lacey", "Lance", "Landon",
    "Lane", "Larry", "Latoya", "Leah", "Levi", "Liam", "Lillian", "Lillie", "Lily", "Lindsay",
    "Lindsey", "Lonnie", "Loretta", "Lori", "Lorraine", "Lucas", "Lucille", "Lucy", "Luis", "Lydia",
    "Lynn", "Mabel", "Mackenzie", "Maddison", "Mae", "Maggie", "Malcolm", "Mallory", "Mandy", "Manuel",
    "Marc", "Marcia", "Marco", "Marcus", "Mariah", "Mario", "Marjorie", "Marsha", "Martin", "Marvin",
    "Mason", "Max", "Maxine", "Maya", "Meagan", "Megan", "Melanie", "Melvin", "Meredith", "Mia",
    "Miguel", "Milton", "Mindy", "Miranda", "Misty", "Mitchell", "Molly", "Monica", "Monique", "Muriel",
    "Myrtle", "Nadia", "Natalie", "Natasha", "Nathaniel", "Neil", "Nelson", "Nina", "Noel", "Nora",
    "Norma", "Norman", "Oscar", "Owen", "Paisley", "Parker", "Pat", "Patty", "Peggy", "Penny",
    "Perry", "Peyton", "Phil", "Priscilla", "Quinn", "Quentin", "Quincy", "Rachael", "Ramon", "Raquel",
    "Reed", "Regina", "Reginald", "Renee", "Reuben", "Rex", "Ricky", "Rita", "Robyn", "Rochelle",
    "Rodney", "Roland", "Roman", "Ron", "Ronnie", "Rosa", "Rosemary", "Ross", "Rowan", "Ruby",
    "Rudy", "Sabrina", "Sadie", "Sally", "Salvador", "Sandy", "Santiago", "Sasha", "Savannah", "Sawyer",
    "Seth", "Shane", "Shannon", "Shelby", "Sheldon", "Shelly", "Sherri", "Sherry", "Sierra", "Simon",
    "Skyler", "Sonia", "Sonya", "Stacey", "Stacy", "Stanley", "Stella", "Stuart", "Sylvia", "Tabitha",
    "Tanner", "Tara", "Ted", "Terrance", "Tiffany", "Tina", "Todd", "Tom", "Tommy", "Toni",
    "Tony", "Tonya", "Tracey", "Traci", "Tracy", "Travis", "Trent", "Tricia", "Troy", "Trudy",
    "Tyrone", "Uma", "Valerie", "Vanessa", "Vera", "Vernon", "Vicki", "Vickie", "Vicky", "Victor",
    "Violet", "Vivian", "Wade", "Wallace", "Wanda", "Warren", "Wendy", "Wesley", "Whitney", "Willie",
    "Willis", "Yolanda", "Yvonne", "Zoe", "Zachariah", "Zane", "Zara", "Zion", "Ada", "Adele"
];

const LAST_NAME_PARTS = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
    "Chen", "Kim", "Patel", "Turner", "Phillips", "Evans", "Parker", "Edwards", "Collins", "Stewart",
    "Morris", "Murphy", "Cook", "Rogers", "Morgan", "Peterson", "Cooper", "Reed", "Bailey", "Bell",
    "Gomez", "Kelly", "Howard", "Ward", "Cox", "Diaz", "Richardson", "Wood", "Watson", "Brooks",
    "Bennett", "Gray", "James", "Reyes", "Cruz", "Hughes", "Price", "Myers", "Long", "Foster",
    "Sanders", "Ross", "Morales", "Powell", "Sullivan", "Russell", "Ortiz", "Jenkins", "Gutierrez", "Perry",
    "Butler", "Barnes", "Fisher", "Henderson", "Coleman", "Simmons", "Patterson", "Jordan", "Reynolds", "Hamilton",
    "Graham", "Shaw", "Gordon", "Wells", "West", "Cole", "Hayes", "Chavez", "Gibson", "Bryant",
    "Ellis", "Stevens", "Murray", "Ford", "Marshall", "McDonald", "Harrison", "Ruiz", "Kennedy", "Wells",
    "Alvarez", "Woods", "Mendoza", "Castillo", "Olson", "Webb", "Washington", "Tucker", "Freeman", "Burns",
    "Henry", "Vasquez", "Snyder", "Simpson", "Crawford", "Jimenez", "Porter", "Mason", "Shaw", "Gordon",
    "Wagner", "Hunter", "Romero", "Hicks", "Dixon", "Hunt", "Palmer", "Robertson", "Black", "Holmes",
    "Stone", "Meyer", "Boyd", "Mills", "Warren", "Fox", "Rose", "Rice", "Moreno", "Schmidt",
    "Patel", "Ferguson", "Nichols", "Herrera", "Medina", "Ryan", "Fernandez", "Weaver", "Daniels", "Stephens",
    "Gardner", "Payne", "Kelley", "Dunn", "Pierce", "Arnold", "Tran", "Spencer", "Peters", "Hawkins",
    "Grant", "Hansen", "Castro", "Hoffman", "Hart", "Elliott", "Cunningham", "Knight", "Bradley", "Carroll",
    "Hudson", "Duncan", "Armstrong", "Berry", "Andrews", "Johnston", "Ray", "Lane", "Riley", "Carpenter",
    "Perkins", "Aguilar", "Silva", "Richards", "Willis", "Matthews", "Chapman", "Lawrence", "Garza", "Vargas",
    "Watkins", "Wheeler", "Larson", "Carlson", "Harper", "George", "Greene", "Burke", "Guzman", "Morrison",
    "Munoz", "Jacobs", "Obrien", "Lawson", "Franklin", "Lynch", "Bishop", "Carr", "Salazar", "Austin",
    "Mendez", "Gilbert", "Jensen", "Williamson", "Montgomery", "Harvey", "Oliver", "Howell", "Dean", "Hanson",
    "Weber", "Garrett", "Sims", "Burton", "Fuller", "Soto", "McCoy", "Welch", "Chen", "Schultz",
    "Walters", "Reid", "Fields", "Walsh", "Little", "Fowler", "Bowman", "Davidson", "May", "Day",
    "Schneider", "Newman", "Brewer", "Lucas", "Holland", "Wong", "Banks", "Santos", "Curtis", "Pearson",
    "Delgado", "Valdez", "Pennington", "Rios", "Douglas", "Sandoval", "Barrett", "Hopkins", "Keller", "Guerrero",
    "Stanley", "Bates", "Alvarado", "Erickson", "Fletcher", "McKinney", "Page", "Dawson", "Joseph", "Marquez",
    "Reeves", "Klein", "Espinoza", "Baldwin", "Moran", "Love", "Robbins", "Higgins", "Ball", "Cortez",
    "Le", "Griffith", "Bowen", "Sharp", "Cummings", "Ramsey", "Hardy", "Swanson", "Barber", "Acosta",
    "Luna", "Chandler", "Blair", "Figueroa", "Dennis", "Oconnor", "Barker", "Logan", "Huffman", "Erickson",
    "Cobb", "Hines", "Barker", "Mullins", "Castaneda", "Maxwell", "Gallegos", "Santana", "Benson", "Rush",
    "McGuire", "Serrano", "Buchanan", "Todd", "Hull", "Gross", "Fitzgerald", "Stokes", "Singleton", "Brock",
    "McDaniel", "McBride", "Oneal", "Landry", "Combs", "Vaughn", "Rasmussen", "Odonnell", "Anthony", "Hull",
    "Barrera", "Oneill", "Morse", "Conner", "Hull", "Barr", "Mcintosh", "Blankenship", "Stark", "Bass",
    "Buckley", "Floyd", "Ritter", "Hanna", "Santos", "McClain", "Newton", "Cantrell", "Barrera", "English",
    "Chung", "Kramer", "Heath", "Hahn", "Middleton", "McLaughlin", "Lam", "Orr", "Jarvis", "McKenzie",
    "Boyer", "McMahon", "Dickerson", "Solomon", "Glenn", "Parsons", "McDowell", "Rush", "Huber", "Morse",
    "Berger", "McKee", "Strickland", "Crane", "Haley", "Barton", "Randolph", "Underwood", "Singleton", "Wilkinson",
    "Duran", "Horne", "Shepherd", "McClure", "Barrera", "Poole", "Calhoun", "Medina", "Vega", "Hodge",
    "McPherson", "Tyler", "McCall", "Sampson", "Briggs", "Hull", "Mullen", "Valencia", "Barr", "Hull",
    "Meadows", "Blackburn", "Dudley", "Nash", "Bruce", "Livingston", "Marks", "Cantrell", "Humphrey", "Dickson",
    "Summers", "Dillon", "Farley", "McKenzie", "Winters", "Branch", "Cherry", "Bass", "Hull", "Barr",
    "Barrera", "Poole", "Calhoun", "Medina", "Vega", "Hodge", "McPherson", "Tyler", "McCall", "Sampson"
];


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

function goToJobsPage() {
    localStorage.setItem("parsedResume", JSON.stringify(resumeJSON));
    window.location.href = "jobs.html";
}