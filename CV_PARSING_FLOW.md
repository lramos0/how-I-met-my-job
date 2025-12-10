# CV Parsing Complete Flow - MeWannaJob System

## System Architecture Overview

The CV parsing system is a **client-side + server-side** hybrid architecture that extracts candidate information from resumes and matches them to jobs using machine learning.

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND: index.html                                │
│                     (Resume Upload & Parsing Page)                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCRIPT.JS: Main Parser Orchestration                   │
│                                                                             │
│  parseResume() function:                                                    │
│  1. User clicks "Analyze & Find Jobs" button                                │
│  2. Get file from input (PDF, DOCX, or TXT)                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
                ▼                      ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │  extractPdfText │    │ extractDocxText │    │   file.text()   │
        │  (using PDF.js) │    │ (using Mammoth) │    │   (for TXT)     │
        └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                 │                      │                      │
                 └──────────────────────┼──────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            restoreWordBoundaries() - Cleanup Extracted Text                │
│  - Splits camelCase words (extractedName → extracted Name)                 │
│  - Separates numbers from letters (2020jobs → 2020 jobs)                   │
│  - Fixes double-spacing issues                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                      ┌────────────────┼────────────────┐
                      │                │                │
                      ▼                ▼                ▼
        ┌──────────────────────┐  ┌──────────────────┐  ┌───────────────────┐
        │  extractSkills()     │  │ extractName()    │  │ extractEducation()│
        │  (Uses SKILL_DICT)   │  │                  │  │ (Uses RegEx)      │
        └──────────┬───────────┘  └────────┬─────────┘  └─────────┬─────────┘
                   │                       │                     │
                   │                       │                     │
        ┌──────────▼───────────┐           │                     │
        │ Comprehensive Skill  │           │                     │
        │ Dictionary Lookup    │           │                     │
        │ (100+ skills with    │           │                     │
        │  aliases & patterns) │           │                     │
        └──────────┬───────────┘           │                     │
                   │                       │                     │
                   │  Returns:             │  Returns:           │  Returns:
                   │  ["Python",           │  "John Doe"         │  "Master" or
                   │   "JavaScript", ...]  │                     │  "Bachelor" or
                   │                       │                     │  "Associate"
                   │                       │                     │
                   └───────────┬───────────┴─────────────────────┘
                               │
                ┌──────────────┼──────────────┬────────────────────────┐
                │              │              │                        │
                ▼              ▼              ▼                        ▼
        ┌─────────────┐ ┌────────────┐ ┌──────────────┐ ┌───────────────────┐
        │extractTitle │ │extractCerts│ │extractLocation    estimateYears()  │
        │             │ │            │ │(Classifies by)    (Parses dates)  │
        │ Finds: Data │ │ Detects:   │ │ - City, State     - Explicit:     │
        │ Engineer,   │ │ - AWS Cert │ │ - ZIP code        "5 years"       │
        │ Developer   │ │ - PMP      │ │ - Country hints   - Date ranges:  │
        │             │ │            │ │ - Indian states   Jan 2020 - Now  │
        └─────────────┘ └────────────┘ └──────────────┘ └───────────────────┘
                                │
                ┌───────────────┼───────────────┬──────────────────────┐
                │               │               │                      │
                ▼               ▼               ▼                      ▼
        ┌────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐
        │extractIndustries│ extractAchievements  extractCerts   (Already done)
        │                │ (Uses Patterns)    │              │
        │ Detects:       │ - Dean's List      │              │
        │ - Education    │ - Scholar          │              │
        │ - Software     │ - Awards/Prizes    │              │
        │ - Healthcare   │ - Competitions     │              │
        └────────────────┘ └──────────────────┘ └──────────────┘ └───────────────┘
                │
                │ All extracted fields combined into:
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   BUILD resumeJSON Object (client-side)                     │
│                                                                              │
│  resumeJSON = {                                                             │
│    inputs: [{                                                               │
│      candidate_id: "CAND-1734000000",                                       │
│      full_name: "John Doe",                                                 │
│      location: "San Diego, CA 92101",                                       │
│      education_level: "Master",                                             │
│      years_experience: 8,                                                   │
│      skills: ["Python", "JavaScript", "AWS"],                               │
│      certifications: ["AWS Solutions Architect"],                           │
│      current_title: "Software Engineer",                                    │
│      industries: ["Software"],                                              │
│      achievements: ["Dean's List", "Published Paper"]                       │
│    }],                                                                      │
│    password: "craig123"                                                     │
│  }                                                                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              sendJSON() - Send to Backend (Netlify Function)                │
│                                                                              │
│  Endpoint: /.netlify/functions/classify-cv                                  │
│  Method: POST                                                               │
│  Body: resumeJSON object (JSON)                                             │
│                                                                              │
│  Purpose: Backend ML model computes competitive_score                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│          BACKEND: Netlify Function (classify-cv/classify-cv.mjs)            │
│                                                                              │
│  1. Receives resumeJSON                                                     │
│  2. Runs Supervised ML Model (Logistic Regression)                         │
│  3. Computes: competitive_score (likelihood of being hired)                │
│  4. Returns: { predictions: [{ competitive_score: 0.87 }] }                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│          SCRIPT.JS: Update Resume with Backend Score                       │
│                                                                              │
│  response = { predictions: [{ competitive_score: 0.87 }] }                │
│  resumeJSON.inputs[0].competitive_score = 0.87                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│        displayParsedResume() - Show Results on index.html                   │
│                                                                              │
│  Shows parsed fields in preview card:                                       │
│  Location: San Diego, CA 92101                                              │
│  Education: Master                                                          │
│  Years Exp: 8                                                               │
│  Title: Software Engineer                                                   │
│  Skills: Python, JavaScript, AWS, ...                                       │
│  Competitive Score: 0.87 (87%)                                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              goToJobsPage() - Navigate to Job Matching                      │
│                                                                              │
│  1. localStorage.setItem("parsedResume", JSON.stringify(resumeJSON))        │
│  2. window.location.href = "jobs.html"                                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   JOBS.HTML + JOBS.JS - Job Matching Page                  │
│                                                                              │
│  loadResume() function on page load:                                        │
│  1. Retrieve resumeJSON from localStorage                                   │
│  2. Normalize skills using SKILL_DICT (alias matching)                     │
│  3. Store resume data in window.resumeData                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│        Fetch Job Data - Load from Netlify Function (get-jobs)               │
│                                                                              │
│  Endpoint: /.netlify/functions/get-jobs                                     │
│  Returns: Array of job postings from database                               │
│  Schema: {                                                                  │
│    job_title, company_name, job_location, job_industries,                  │
│    job_seniority_level, job_employment_type, job_posted_date,              │
│    job_description, salary, ...                                            │
│  }                                                                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              populateFilters() - Build Filter Dropdowns                     │
│                                                                              │
│  From job data, extract unique values:                                      │
│  - Companies: Apple, Google, Microsoft, etc.                                │
│  - Locations: San Diego, New York, Remote, etc.                             │
│  - Industries: Software, Finance, Healthcare, etc.                          │
│  - Seniority: Junior, Senior, Lead, Executive, etc.                         │
│  - Employment Type: Full-time, Part-time, Contract, etc.                    │
│                                                                              │
│  Populate select elements: #filterCompany, #filterLocation, etc.            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            filterJobs() - Main Matching & Filtering Logic                   │
│                                                                              │
│  For each job posting:                                                      │
│                                                                              │
│  1. SKILL MATCHING                                                          │
│     - Extract skills from job description                                   │
│     - Compare with candidate's skills                                       │
│     - Compute skill_overlap_score (0-1)                                    │
│     - Example: 8 of 12 required skills = 0.67                              │
│                                                                              │
│  2. RELEVANCE SCORING (Text-based)                                         │
│     - TF-IDF of resume vs job description                                  │
│     - Keyword matching (years exp, education, title)                       │
│     - Location proximity scoring                                            │
│     - Result: relevance_score (0-1)                                        │
│                                                                              │
│  3. COMBINED MATCHING SCORE                                                │
│     final_score = 0.6 * skill_overlap + 0.4 * relevance_score             │
│                                                                              │
│  4. APPLY USER FILTERS                                                      │
│     - Company filter                                                        │
│     - Location filter                                                       │
│     - Industry filter                                                       │
│     - Seniority level filter                                                │
│     - Employment type filter                                                │
│     - Global search (title/company/keywords)                                │
│     - Date filter (posted after date)                                       │
│                                                                              │
│  5. SORT RESULTS                                                            │
│     - By relevance (descending)                                             │
│     - By match score (high/low)                                             │
│     - By date (newest first)                                                │
│     - By salary (highest first)                                             │
│                                                                              │
│  6. RETURN FILTERED & SORTED JOBS                                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            renderJobs() - Display Matched Jobs on Page                      │
│                                                                              │
│  For each matched job, create card showing:                                 │
│  - Job Title & Company                                                      │
│  - Match Score (visual bar: 0-100%)                                         │
│  - Required Skills (with icons)                                             │
│  - Candidate's Matching Skills (highlighted in green)                       │
│  - Missing Skills (highlighted in red)                                      │
│  - Location & Seniority                                                     │
│  - Employment Type                                                          │
│  - Salary & Posted Date                                                     │
│  - View Details button                                                      │
│  - Apply button                                                             │
│                                                                              │
│  Result Count Badge: "150 found" (total matches)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│          User Interaction - View & Filter Results                           │
│                                                                              │
│  User can:                                                                   │
│  1. Click job card to view full job description                             │
│  2. Apply filters and search in real-time                                   │
│  3. Sort by different criteria                                              │
│  4. Click "New Resume" to upload different resume                           │
│  5. Click "Apply" (external link to job posting)                            │
│                                                                              │
│  Sticky sidebar shows:                                                      │
│  - Candidate profile summary                                                │
│  - Competitive score                                                        │
│  - Inferred job type preferences                                            │
│  - Skills breakdown                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. **index.html** - Resume Upload Page
- **Purpose**: Entry point for users
- **Key Elements**:
  - File input for resume (PDF, DOCX, TXT)
  - "Analyze & Find Jobs" button
  - Status messages display
  - Preview section (hidden initially, shows parsed resume)
  - Script includes:
    - `skill_dict.js` - Comprehensive skill dictionary
    - `education_text_handler.js` - Education level classifier
    - `achievementClassifier.js` - Achievement/award parser
    - `locationClassifier.js` - Location extractor
    - `script.js` - Main parsing logic

---

### 2. **script.js** - Main Parser Orchestration

#### **Text Extraction Functions**:
```javascript
extractPdfText(file)
  ├─ Uses: PDF.js library
  ├─ Method: Converts PDF to text with line break detection
  └─ Returns: Cleaned text string

extractDocxText(file)
  ├─ Uses: Mammoth.js library
  └─ Returns: Raw text from DOCX

file.text()
  └─ For plain TXT files (native JS API)
```

#### **Text Cleanup**:
```javascript
restoreWordBoundaries(text)
  ├─ Splits camelCase: extractedName → extracted Name
  ├─ Separates numbers: 2020job → 2020 job
  ├─ Fixes spacing issues
  └─ Returns: Cleaned text
```

#### **Field Extraction Functions**:

| Function | Input | Output | Method |
|----------|-------|--------|--------|
| `extractSkills(text)` | Resume text | String[] of skills | SKILL_DICT alias lookup |
| `extractName(text)` | Resume text | String name | Regex: `[A-Z][a-z]+ [A-Z][a-z]+` |
| `extractEducation(text)` | Resume text | "Doctorate"\|"Master"\|"Bachelor"\|"Associate" | Keyword detection (using education_text_handler.js) |
| `estimateYears(text)` | Resume text | Number | Date range parsing + explicit "X years" mentions |
| `extractLocation(text)` | Resume text | String location | City/State/Country detection (using locationClassifier.js) |
| `extractTitle(text)` | Resume text | String role | Keyword matching against predefined roles |
| `extractCerts(text)` | Resume text | String[] | Regex patterns for known certifications |
| `extractIndustries(text)` | Resume text | String[] | Keyword-based industry detection |
| `extractAchievements(text)` | Resume text | String[] | Pattern matching for honors, awards, publications |

---

### 3. **model/data/skill_dict.js** - Comprehensive Skill Dictionary

**Structure**:
```javascript
const SKILL_DICT = {
  "Canonical Skill Name": ["alias1", "alias2", "pattern", ...],
  "Python": ["python", "python3", "cpython", "pypy"],
  "SQL": ["\bsql\b", "mysql", "postgresql", "postgres", "oracle", "sqlite"],
  "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
  // ... ~100 more skills
};
```

**Features**:
- **100+ canonical skills** across categories:
  - Programming Languages (Python, Java, C++, R, SQL)
  - Frameworks (React, Django, Spring, Flask)
  - Cloud Platforms (AWS, Azure, GCP)
  - Data Tools (TensorFlow, PyTorch, Spark, Hadoop)
  - DevOps (Docker, Kubernetes, Jenkins, GitLab CI)
  - Databases (MongoDB, PostgreSQL, Redis, Elasticsearch)
  - BI Tools (Tableau, Power BI, Looker, Qlik)
  - Design (Figma, Sketch, Adobe XD)
  - Project Management (Agile, Scrum, PMP)

- **Alias Matching**: Single skill can match multiple keywords
  - "React" matches: react, reactjs, react.js
  - "Scikit-Learn" matches: sklearn, scikit-learn, scikit
  - "SQL" matches: sql, mysql, postgresql, oracle, sqlite (with word boundaries)

---

### 4. **model/parsing/education_text_handler.js** - Education Classifier

**Detects** (in priority order):
1. **Doctorate** (~15 patterns):
   - PhD/Ph.D., Doctorate, DPhil
   - JD (Law), DO/DDS/DMS/DNP (Medical)
   - PharmD, Psy.D.

2. **Master** (~10 patterns):
   - Master's, MS/M.S., MA/M.A., MSC, MBA
   - M.Eng, M.Ed, MSN, Nurse Practitioner
   - Negative: Excludes "headmaster", "master plan", "master electrician"

3. **Bachelor** (~10 patterns):
   - Bachelor's, BS/B.S., BA/B.A., BSC, BFA
   - B.Eng., BE (Engineering)
   - BSN (Nursing)

4. **Associate** (~5 patterns):
   - Associate's, AA, AS, AAS

5. **Default**: Returns education level found, or "Unknown"

---

### 5. **model/parsing/locationClassifier.js** - Location Extractor

**Detection Strategy** (priority order):

1. **City, State, ZIP** (Most specific)
   - Pattern: `City, ST 12345`
   - Example: `San Diego, CA 92101`

2. **US State Names** (50 states + DC)
   - Matches full names: "California", "Texas"
   - Returns with ZIP if found nearby

3. **Indian States** (36 states/UTs)
   - Matches: "Maharashtra", "Karnataka", "Delhi"
   - Returns with PIN (6-digit code) if found

4. **US ZIP codes** (5 or 9 digits)
   - Pattern: `12345` or `12345-6789`

5. **India PIN codes** (6 digits, with optional space)
   - Pattern: `123456` or `123 456`

6. **Country Detection**
   - Looks for hints: USA, UK, Canada, Australia, India, etc.

**Output Example**:
- Resume has: "San Diego, California 92101"
- Returns: `"San Diego, CA 92101"`

---

### 6. **model/parsing/achievementClassifier.js** - Achievement Extractor

**Detects** (multiple patterns):

1. **Academic Honors**:
   - Dean's List, Chancellor's List
   - Summa Cum Laude, Magna Cum Laude, Cum Laude

2. **Scholarships & Fellowships**:
   - Scholarship, Fellowship mentions

3. **Publications & Awards**:
   - Best Paper, Best Poster
   - Publication Award

4. **Competitions & Contests**:
   - Hackathon, Case Competition, Datathon
   - CodeFest, ICPC, ACM, Olympiad
   - Science Fair, Robotics
   - Kaggle, LeetCode, Codeforces

5. **Rankings & Placements**:
   - 1st Place, Runner-up, Finalist
   - Top X% performers

**Output Example**:
```javascript
[
  "Dean's List",
  "Won first place in case competition 2023",
  "Hackathon finalist at TechCrunch Disrupt"
]
```

---

### 7. **resumeJSON Structure** (Client-side Object)

```javascript
{
  "inputs": [
    {
      "candidate_id": "CAND-1734000000",           // Unique ID (timestamp-based)
      "full_name": "John Doe",                     // From regex extraction
      "location": "San Diego, CA 92101",           // From locationClassifier
      "education_level": "Master",                 // From education_text_handler
      "years_experience": 8,                       // From date range parsing
      "skills": ["Python", "AWS", "Docker"],       // From SKILL_DICT matching
      "certifications": ["AWS Solutions Architect"], // From extractCerts()
      "current_title": "Software Engineer",        // From role keywords
      "industries": ["Software"],                  // From industry keywords
      "achievements": [                            // From achievementClassifier
        "Dean's List",
        "Published research paper"
      ]
    }
  ],
  "password": "craig123"                          // Backend auth
}
```

---

### 8. **Backend Communication** - Netlify Functions

#### **sendJSON() - POST to classify-cv**
```
Endpoint: /.netlify/functions/classify-cv
Method: POST
Body: resumeJSON (all parsed fields)
Response: {
  "predictions": [{
    "competitive_score": 0.87  // 0-1 range
  }]
}
```

**Backend Processing**:
1. Receives resumeJSON
2. Loads trained Logistic Regression model (from applicant-evaluator-training.py)
3. Vectorizes candidate features using HashingTF
4. Computes probability of candidate being hired
5. Returns competitive_score

---

### 9. **jobs.html** - Job Matching Dashboard

**Layout**:
```
┌─────────────────────────────────────────────────────────┐
│ Navbar: MeWannaJob | New Resume button                  │
├────────┬────────────────────────────────────────────────┤
│        │                                                │
│ Sidebar│        Main Content Area                       │
│ (3col) │ (9col)                                         │
│        │                                                │
│        │  Search & Filter Toolbar                       │
│ Profile│  ┌─────────────────────────────────────────┐   │
│ Summary│  │ [Search Box] [Search Button]            │   │
│        │  │ [Company▼] [Location▼] [Industry▼]      │   │
│        │  │ [Seniority▼] [Type▼] [Sort▼]            │   │
│        │  │ [Date Filter] [Apply Filters] [Reset]   │   │
│        │  └─────────────────────────────────────────┘   │
│        │                                                │
│        │  Job Results (Infinite scroll / paginated)     │
│ Job    │  ┌──────────────────────────────────────┐      │
│ Type   │  │ Job Card 1: Engineer @ Google        │      │
│ Insights│ │ Match: ████████░░ 82%                │      │
│        │  │ Skills: Python✓ Java✓ SQL✗ Docker✓ │       │
│        │  └──────────────────────────────────────┘      │
│        │                                                │
│        │  ┌──────────────────────────────────────┐      │
│        │  │ Job Card 2: Data Scientist @ Meta    │      │
│        │  │ Match: █████░░░░░░ 51%               │      │
│        │  │ Skills: Python✓ R✗ TensorFlow✓...   │      │
│        │  └──────────────────────────────────────┘      │
│        │                                                │
└────────┴────────────────────────────────────────────────┘
```

---

### 10. **jobs.js** - Job Matching Engine

#### **Main Functions**:

```javascript
loadResume()
  ├─ Retrieve from localStorage
  ├─ Parse resumeJSON structure
  ├─ Normalize skills using SKILL_DICT aliases
  ├─ Display profile summary in sidebar
  ├─ Fetch jobs from get-jobs endpoint
  └─ Call populateFilters() and filterJobs()

populateFilters()
  ├─ Extract unique companies, locations, industries, seniority, types
  └─ Populate <select> dropdowns with unique values

filterJobs()
  ├─ For each job:
  │  ├─ Extract required skills from description
  │  ├─ Compute skill_overlap_score
  │  ├─ Compute relevance_score (TF-IDF, keywords, location)
  │  ├─ Compute final_score = 0.6 * skills + 0.4 * relevance
  │  └─ Apply user filters (company, location, industry, etc.)
  │
  ├─ Sort results (relevance, score, date, salary)
  └─ Call renderJobs()

renderJobs(jobsFiltered)
  ├─ Clear previous results
  ├─ For each job, create card:
  │  ├─ Title, Company, Match Score bar
  │  ├─ Skills section (matching highlighted green, missing red)
  │  ├─ Location, Seniority, Employment Type
  │  ├─ Salary & Posted Date
  │  └─ View Details & Apply buttons
  └─ Update result count badge

clearFilters()
  └─ Reset all filters and re-run filterJobs()

inferJobTypes(resume)
  └─ Analyze candidate's skills/experience to suggest job types
```

---

### 11. **Matching Algorithm** - Core Logic

```javascript
// SKILL MATCHING
skill_overlap_score = (matching_skills / required_skills) * 1.0
// Example: Candidate has 8 of 12 required skills = 0.67

// RELEVANCE SCORING (Text-based)
relevance_score = computed from:
  ├─ TF-IDF similarity (resume vs job description)
  ├─ Years experience match (candidate vs required)
  ├─ Education level match (candidate vs required)
  ├─ Location proximity (exact match = 1.0)
  ├─ Title relevance (role matching)
  └─ Industry alignment

// COMBINED SCORE
final_score = 0.6 * skill_overlap_score + 0.4 * relevance_score
// Result: 0-1 (displayed as 0-100% match)

// FILTERING (After scoring)
show job if:
  ├─ Company matches filter OR "any" selected
  ├─ Location matches filter OR "any" selected
  ├─ Industry matches filter OR "any" selected
  ├─ Seniority matches filter OR "any" selected
  ├─ Employment type matches filter OR "any" selected
  ├─ Global search matches title/company/keywords
  ├─ Posted date >= selected date filter
  └─ (final_score > threshold)
```

---

## Data Flow Summary

```
Resume File (PDF/DOCX/TXT)
        │
        ▼
Text Extraction + Cleanup
        │
        ├─► extractSkills (SKILL_DICT)
        ├─► extractEducation (regex + patterns)
        ├─► estimateYears (date parsing)
        ├─► extractLocation (city/state/country)
        ├─► extractTitle (role keywords)
        ├─► extractName (regex)
        ├─► extractCerts (pattern matching)
        ├─► extractIndustries (keywords)
        └─► extractAchievements (awards/honors)
        │
        ▼
resumeJSON Object (client-side)
        │
        ├─► Display on index.html (preview)
        │
        ├─► Send to Backend (classify-cv)
        │   └─► Returns competitive_score
        │
        └─► Store in localStorage
            │
            ▼
        jobs.html loads page
            │
            ├─► Retrieve from localStorage
            ├─► Normalize skills (aliases)
            ├─► Display in sidebar (profile summary)
            │
            ├─► Fetch jobs (get-jobs endpoint)
            │
            └─► For each job:
                ├─► Score: skill_overlap + relevance
                ├─► Filter: apply user filters
                ├─► Sort: by relevance/score/date/salary
                └─► Display: rendered job cards

```

---

## Key Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF parsing | PDF.js | Extract text from PDF resumes |
| DOCX parsing | Mammoth.js | Extract text from Word documents |
| Education detection | Regex + keyword patterns | Classify education level |
| Location detection | Regex + geography database | Extract city/state/country |
| Skill matching | SKILL_DICT (100+ skills) | Match skills with aliases |
| Achievement detection | Pattern matching | Detect awards, honors, competitions |
| Backend ML | Logistic Regression (PySpark) | Compute competitive score |
| Frontend framework | Bootstrap 5 | UI/UX layout and components |
| Job matching | TF-IDF + skill overlap | Score and rank job matches |
| Storage | localStorage | Persist parsed resume (client-side) |

---

## Summary

The CV parsing system follows this complete pipeline:

1. **User uploads resume** (index.html)
2. **Client-side parsing** (script.js) extracts all fields using specialized extractors
3. **Backend scoring** (Netlify function) computes competitive_score using ML
4. **Resume preview** displays parsed fields on index.html
5. **User navigates to jobs.html** with parsed resume in localStorage
6. **Job matching engine** (jobs.js) scores and filters jobs based on:
   - Skill overlap (using SKILL_DICT)
   - Relevance scoring (TF-IDF, keywords, location)
   - User-selected filters
7. **Job cards rendered** with match scores, skill highlights, and filtering options
8. **User can explore, filter, and apply** to matched jobs

This architecture separates concerns (parsing vs matching vs UI) while leveraging ML for intelligent candidate evaluation.
