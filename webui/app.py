import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import re
import spacy
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from datetime import datetime

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define comprehensive skill patterns
SKILL_PATTERNS = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "PHP", "Swift", "Go", "Rust",
    "Kotlin", "Scala", "R", "MATLAB", "Perl", "Shell", "Bash", "PowerShell", "Assembly",
    
    # Web Technologies - Frontend
    "HTML", "HTML5", "CSS", "CSS3", "SASS", "SCSS", "Less",
    "React", "Angular", "Vue.js", "Svelte", "Next.js", "Nuxt.js", "jQuery",
    "Redux", "MobX", "Vuex", "WebGL", "Three.js", "D3.js", "Material-UI", "Tailwind CSS",
    "Bootstrap", "WebAssembly", "PWA", "Service Workers",
    
    # Web Technologies - Backend
    "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot", "Laravel",
    "ASP.NET", "Ruby on Rails", "PHP", "Symfony", "GraphQL", "REST API", "WebSocket",
    "gRPC", "Microservices", "SOA", "OAuth", "JWT", "SOAP",
    
    # Data Science & Analytics
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Neural Networks",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost", "LightGBM",
    "Data Analysis", "Data Mining", "Data Visualization", "Statistical Analysis",
    "Big Data", "Hadoop", "Spark", "Data Warehouse", "ETL", "Data Pipeline",
    "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly", "SPSS", "SAS",
    
    # Cloud & DevOps
    "AWS", "Azure", "Google Cloud", "IBM Cloud", "Oracle Cloud", "DigitalOcean",
    "Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions", "Travis CI",
    "Terraform", "Ansible", "Puppet", "Chef", "CI/CD", "Infrastructure as Code",
    "Cloud Architecture", "Serverless", "Lambda", "ECS", "EKS", "S3", "EC2", "RDS",
    
    # Databases & Storage
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQL Server", "SQLite",
    "Redis", "Elasticsearch", "Cassandra", "DynamoDB", "Neo4j", "MariaDB",
    "Firebase", "Supabase", "Data Modeling", "Database Design", "NoSQL",
    
    # Tools & Development
    "Git", "SVN", "Mercurial", "JIRA", "Confluence", "Trello", "Asana",
    "VS Code", "IntelliJ IDEA", "Eclipse", "Visual Studio", "Vim",
    "Webpack", "Babel", "ESLint", "Jest", "Mocha", "Cypress", "Selenium",
    "Unit Testing", "Integration Testing", "TDD", "BDD", "Code Review",
    
    # Mobile Development
    "iOS", "Android", "React Native", "Flutter", "Xamarin", "Unity",
    "Swift UI", "Kotlin Multiplatform", "Mobile Architecture",
    "App Store", "Google Play", "Mobile Security", "Push Notifications",
    
    # Security
    "Cybersecurity", "Network Security", "Encryption", "OAuth", "SAML",
    "Penetration Testing", "Security Auditing", "Firewall", "VPN",
    "Authentication", "Authorization", "SSO", "Identity Management",
    
    # Methodologies & Practices
    "Agile", "Scrum", "Kanban", "Lean", "Waterfall", "XP", "SAFe",
    "Project Management", "Product Management", "Technical Leadership",
    "System Design", "Software Architecture", "Design Patterns", "SOLID",
    
    # Soft Skills & Business
    "Team Leadership", "Communication", "Problem Solving", "Critical Thinking",
    "Collaboration", "Time Management", "Stakeholder Management",
    "Strategic Planning", "Business Analysis", "Requirements Gathering",
    "Technical Writing", "Public Speaking", "Mentoring", "Cross-functional Leadership",
    
    # Emerging Technologies
    "Blockchain", "Smart Contracts", "Solidity", "Web3", "NFT",
    "AR/VR", "Artificial Intelligence", "IoT", "5G", "Edge Computing",
    "Quantum Computing", "Robotics", "RPA", "Digital Transformation",
    
    # Quality & Performance
    "Performance Optimization", "Load Testing", "Stress Testing",
    "Quality Assurance", "Quality Control", "Debugging", "Profiling",
    "Code Optimization", "Memory Management", "Concurrency",
    
    # Industry Standards & Compliance
    "GDPR", "HIPAA", "SOX", "PCI DSS", "ISO 27001",
    "Regulatory Compliance", "Data Privacy", "Risk Management",
    "Change Management", "Release Management"
]

# Load job data
@st.cache_data
def load_job_data():
    return pd.read_csv(r"C:\Users\takbh\OneDrive\Desktop\BDA696\project\how-I-met-my-job\webui\data\job_postings.csv")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from uploaded PDF resume
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract skills from text using spaCy and predefined skill patterns
def extract_skills(text, nlp, skill_patterns):
    doc = nlp(text.lower())
    skills = set()
    
    # Extract skills using pattern matching
    for skill in skill_patterns:
        if skill.lower() in text.lower():
            skills.add(skill)
    
    # Extract technical terms and proper nouns that might be skills
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT']:  # Organizations and products are often technologies/tools
            skills.add(ent.text)
            
    return list(skills)

# Extract years of experience from text
def extract_experience_details(text):
    # Patterns for different experience formats
    patterns = [
        r'(\d+)\s*\+?\s*years?(?:\s+of)?\s+(?:of\s+)?experience',
        r'(\d+)\s*\+?\s*yrs?(?:\s+of)?\s+(?:of\s+)?experience',
        r'experience\s*(?:of|for|:)?\s*(\d+)\s*\+?\s*years?',
        r'experience\s*(?:of|for|:)?\s*(\d+)\s*\+?\s*yrs?'
    ]
    
    max_years = 0
    for pattern in patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            years = int(match.group(1))
            max_years = max(max_years, years)
    
    return max_years

# Compute weighted similarity scores considering multiple factors
def compute_similarity(resume_text, jobs_df, model, skill_patterns):
    # Initialize weights for different matching criteria
    weights = {
        'description_similarity': 0.3,
        'skills_match': 0.3,
        'title_similarity': 0.2,
        'experience_match': 0.2
    }
    
    # Process resume
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    resume_skills = extract_skills(resume_text, nlp, skill_patterns)
    resume_experience = extract_experience_details(resume_text)
    
    # Initialize results
    total_scores = []
    
    # Process each job
    for _, job in jobs_df.iterrows():
        # Description similarity (semantic matching)
        job_embedding = model.encode(job['job_summary'], convert_to_tensor=True)
        description_score = float(util.cos_sim(resume_embedding, job_embedding)[0][0])
        
        # Skills matching
        job_skills = extract_skills(job['job_summary'], nlp, skill_patterns)
        if job_skills and resume_skills:
            skills_score = len(set(resume_skills) & set(job_skills)) / len(set(job_skills))
        else:
            skills_score = 0
            
        # Title similarity
        title_embedding = model.encode(job['job_title'], convert_to_tensor=True)
        title_score = float(util.cos_sim(resume_embedding, title_embedding)[0][0])
        
        # Experience matching
        job_required_exp = extract_experience_details(job['job_summary'])
        if job_required_exp > 0 and resume_experience > 0:
            exp_score = min(1.0, resume_experience / job_required_exp) if job_required_exp > 0 else 0.5
        else:
            exp_score = 0.5  # Neutral score if experience requirements are unclear
        
        # Calculate weighted total score
        total_score = (
            weights['description_similarity'] * description_score +
            weights['skills_match'] * skills_score +
            weights['title_similarity'] * title_score +
            weights['experience_match'] * exp_score
        )
        
        total_scores.append(total_score)
    
    return np.array(total_scores)

# Load data and model
df = load_job_data()
model = load_model()

# Streamlit UI
st.set_page_config(
    page_title="How I Met My JOB",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced visual styling with tooltips
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Tooltip Styling */
    .tooltip-container {
        position: relative;
        display: inline-block;
    }
    
    .tooltip-container .tooltip-text {
        visibility: hidden;
        background-color: #2c3e50;
        color: white;
        text-align: left;
        padding: 8px 12px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        width: 220px;
        font-size: 14px;
        line-height: 1.4;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    .tooltip-container .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #2c3e50 transparent transparent transparent;
    }
    
    /* Info Icon */
    .info-icon {
        color: #6c757d;
        font-size: 16px;
        margin-left: 5px;
        cursor: help;
    }
    
    /* Match Percentage Styling */
    .match-percentage {
        font-size: 24px;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 15px;
        margin: 5px;
        text-align: center;
    }
    .match-excellent {
        background-color: #28a745;
        color: white;
    }
    .match-good {
        background-color: #17a2b8;
        color: white;
    }
    .match-fair {
        background-color: #ffc107;
        color: black;
    }
    .match-poor {
        background-color: #dc3545;
        color: white;
    }
    
    /* Skill Tag Styling */
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 20px;
        font-size: 14px;
    }
    .skill-match {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #2e7d32;
    }
    .skill-missing {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #c62828;
    }
    
    /* Job Card Styling */
    .job-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Progress Bar Styling */
    .match-progress {
        width: 100%;
        height: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
        overflow: hidden;
        margin: 5px 0;
    }
    .match-progress-bar {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 14px;
        color: #5f6368;
    }
    
    /* Apply Button Styling */
    .apply-button {
        background-color: #1a73e8;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        border-radius: 25px;
        margin-top: 15px;
        width: 100%;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .apply-button:hover {
        background-color: #1557b0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 How I Met My JOB ^-^")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;'>
        <h4>Welcome to your job matching assistant!</h4>
        <p>Upload your resume and let AI find the best matching jobs for you. 
        Use the filters in the sidebar to refine your search.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("📋 Filter Jobs")

company = st.sidebar.selectbox("Company Name", ["All"] + sorted(df['company_name'].dropna().unique()))
title = st.sidebar.selectbox("Job Title", ["All"] + sorted(df['job_title'].dropna().unique()))
location = st.sidebar.selectbox("Job Location", ["All"] + sorted(df['job_location'].dropna().unique()))
country = st.sidebar.selectbox("Country Code", ["All"] + sorted(df['country_code'].dropna().unique()))
seniority = st.sidebar.selectbox("Seniority Level", ["All"] + sorted(df['job_seniority_level'].dropna().unique()))
employment_type = st.sidebar.selectbox("Employment Type", ["All"] + sorted(df['job_employment_type'].dropna().unique()))
industry = st.sidebar.selectbox("Industry", ["All"] + sorted(df['job_industries'].dropna().unique()))
salary_range = st.sidebar.text_input("Base Pay Range (e.g. 50000-100000)")
date_range = st.sidebar.date_input("Posted Date Range", [])

keyword = st.sidebar.text_input("Keyword Search in Job Summary")

# Apply filters
filtered_df = df.copy()
if company != "All":
    filtered_df = filtered_df[filtered_df['company_name'] == company]
if title != "All":
    filtered_df = filtered_df[filtered_df['job_title'] == title]
if location != "All":
    filtered_df = filtered_df[filtered_df['job_location'] == location]
if country != "All":
    filtered_df = filtered_df[filtered_df['country_code'] == country]
if seniority != "All":
    filtered_df = filtered_df[filtered_df['job_seniority_level'] == seniority]
if employment_type != "All":
    filtered_df = filtered_df[filtered_df['job_employment_type'] == employment_type]
if industry != "All":
    filtered_df = filtered_df[filtered_df['job_industries'] == industry]
if salary_range:
    try:
        min_salary, max_salary = map(int, salary_range.split('-'))
        filtered_df = filtered_df[filtered_df['job_base_pay_range'].apply(
            lambda x: isinstance(x, str) and any(char.isdigit() for char in x) and min_salary <= int(''.join(filter(str.isdigit, x))) <= max_salary)]
    except:
        st.warning("Invalid salary range format. Use format like 50000-100000.")
if date_range:
    try:
        filtered_df['job_posted_date'] = pd.to_datetime(filtered_df['job_posted_date'], errors='coerce')
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[(filtered_df['job_posted_date'] >= pd.to_datetime(start_date)) & (filtered_df['job_posted_date'] <= pd.to_datetime(end_date))]
    except:
        st.warning("Invalid date format in job_posted_date column.")
if keyword:
    filtered_df = filtered_df[filtered_df['job_summary'].str.contains(keyword, case=False, na=False)]

# Resume upload section
st.markdown("### 📄 Upload Your Resume")
st.markdown("Upload your resume in PDF format to find matching jobs.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file:
    try:
        with st.spinner('Reading and analyzing your resume...'):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text.strip():
                st.error("Could not extract text from the PDF. Please make sure the PDF contains readable text.")
            else:
                st.success("Resume uploaded and parsed successfully!")
                
                # Show resume preview
                with st.expander("Preview extracted text"):
                    st.text_area("Resume Content", resume_text, height=200)
    except Exception as e:
        st.error(f"Error processing the resume: {str(e)}")
        st.info("Please try uploading a different PDF file.")

    # Compute match scores with detailed analysis
    with st.spinner('Analyzing resume and computing job matches...'):
        match_scores = compute_similarity(resume_text, filtered_df, model, SKILL_PATTERNS)
        filtered_df['match_score'] = match_scores
        
        # Extract skills from resume for display
        resume_skills = extract_skills(resume_text, nlp, SKILL_PATTERNS)
        resume_experience = extract_experience_details(resume_text)
        
        # Sort and get top matches
        top_matches = filtered_df.sort_values(by='match_score', ascending=False).head(10)
        
        # Display resume analysis
        st.subheader("📊 Resume Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🛠️ Skills Identified**")
            if resume_skills:
                for skill in sorted(resume_skills):
                    st.markdown(f"- {skill}")
            else:
                st.info("No specific skills were identified. Consider adding more technical terms to your resume.")
                
        with col2:
            st.markdown("**👨‍💼 Experience Level**")
            if resume_experience > 0:
                st.markdown(f"**{resume_experience}** years of experience detected")
            else:
                st.info("No clear experience duration found. Consider adding your years of experience explicitly.")

    # Display match overview with enhanced styling
    st.markdown("## 🎯 Job Matching Results")
    
    # Helper function to get match class
    def get_match_class(score):
        if score >= 0.8: return "excellent"
        elif score >= 0.6: return "good"
        elif score >= 0.4: return "fair"
        else: return "poor"
    
    # Add a metric row for overall statistics with custom styling
    st.markdown("""
        <div style='display: flex; justify-content: space-between; margin: 20px 0;'>
    """, unsafe_allow_html=True)
    
    metrics = [
        {
            "label": "Top Match Score",
            "value": f"{max(match_scores):.0%}",
            "icon": "🎯",
            "tooltip": "The highest match percentage found among all job listings. This considers skills, experience, and job description relevance."
        },
        {
            "label": "Average Match",
            "value": f"{np.mean(match_scores):.0%}",
            "icon": "📊",
            "tooltip": "The average match score across all job listings. This gives you an idea of how well your profile matches the overall job market."
        },
        {
            "label": "Skills Identified",
            "value": f"{len(resume_skills)}",
            "icon": "🛠️",
            "tooltip": "The number of technical and soft skills extracted from your resume. More identified skills can lead to better matches."
        },
        {
            "label": "Experience",
            "value": f"{resume_experience} years",
            "icon": "⏳",
            "tooltip": "Total years of professional experience detected in your resume. This is used to match against job requirements."
        }
    ]
    
    for metric in metrics:
        st.markdown(f"""
            <div class='metric-card tooltip-container'>
                <div class='metric-value'>{metric['icon']} {metric['value']}</div>
                <div class='metric-label'>
                    {metric['label']}
                    <span class='info-icon'>ℹ️</span>
                </div>
                <span class='tooltip-text'>{metric['tooltip']}</span>
            </div>
        """, unsafe_allow_html=True)
    
    # Display top matches with detailed analysis
    for idx, job in top_matches.iterrows():
        # Calculate detailed match scores
        job_skills = extract_skills(job['job_summary'], nlp, SKILL_PATTERNS)
        skills_matched = set(resume_skills) & set(job_skills)
        skills_missing = set(job_skills) - set(resume_skills)
        skill_match_pct = len(skills_matched) / len(job_skills) if job_skills else 0
        
        # Create match score breakdown
        description_score = float(util.cos_sim(model.encode(resume_text), model.encode(job['job_summary']))[0][0])
        title_score = float(util.cos_sim(model.encode(resume_text), model.encode(job['job_title']))[0][0])
        
        # Calculate experience match
        job_exp = extract_experience_details(job['job_summary'])
        exp_match = min(1.0, resume_experience / job_exp) if job_exp > 0 else 0.5

        with st.expander(
            f"📌 {job['job_title']} at {job['company_name']} - Overall Match: {job['match_score']:.0%}",
            expanded=(idx == 0)  # Expand first result by default
        ):
            # Match Score Breakdown with enhanced styling and tooltips
            st.markdown("### 📊 Match Score Breakdown")
            
            # Create progress bars for each score component with tooltips
            score_components = [
                {
                    "label": "Skills Match",
                    "score": skill_match_pct,
                    "tooltip": f"""
                        Skills Match Score: {skill_match_pct:.0%}
                        • Matched Skills: {len(skills_matched)}
                        • Required Skills: {len(job_skills)}
                        • Missing Skills: {len(skills_missing)}
                        
                        This score shows how many of the job's required skills appear in your resume.
                        Higher scores mean you have more of the specific skills the job requires.
                    """
                },
                {
                    "label": "Description Match",
                    "score": description_score,
                    "tooltip": f"""
                        Description Relevance: {description_score:.0%}
                        
                        This score measures how well your overall experience aligns with the job description.
                        It uses AI to compare your resume's content with the job requirements and responsibilities.
                        
                        Factors considered:
                        • Technical terminology matches
                        • Role responsibilities alignment
                        • Industry-specific knowledge
                    """
                },
                {
                    "label": "Title Relevance",
                    "score": title_score,
                    "tooltip": f"""
                        Title Match Score: {title_score:.0%}
                        
                        This indicates how well your experience matches the job title.
                        Based on:
                        • Current/past job titles
                        • Role seniority
                        • Position type alignment
                    """
                },
                {
                    "label": "Experience Match",
                    "score": exp_match,
                    "tooltip": f"""
                        Experience Match: {exp_match:.0%}
                        
                        Your Experience: {resume_experience} years
                        Job Requires: {job_exp if job_exp > 0 else 'Not specified'} years
                        
                        This score compares your years of experience with the job's requirements.
                        A score of 100% means you meet or exceed the required experience level.
                    """
                }
            ]
            
            for component in score_components:
                match_class = get_match_class(component["score"])
                st.markdown(f"""
                    <div class='tooltip-container' style='margin: 10px 0;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                            <span>
                                {component["label"]}
                                <span class='info-icon'>ℹ️</span>
                            </span>
                            <span class='match-percentage match-{match_class}'>{component["score"]:.0%}</span>
                        </div>
                        <div class='match-progress'>
                            <div class='match-progress-bar match-{match_class}' 
                                 style='width: {component["score"]:.0%}; 
                                        background-color: {
                                            "#28a745" if match_class == "excellent"
                                            else "#17a2b8" if match_class == "good"
                                            else "#ffc107" if match_class == "fair"
                                            else "#dc3545"
                                        };'>
                            </div>
                        </div>
                        <span class='tooltip-text' style='white-space: pre-line;'>{component["tooltip"]}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Main content columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📝 Job Summary")
                st.write(job['job_summary'])
                
                # Skills Analysis
                st.markdown("### 🛠️ Skills Analysis")
                
                # Compact Skills Analysis with tabs
                skill_tabs = st.tabs(["📊 Skills Overview", "✅ Matched Skills", "📚 Missing Skills"])
                
                with skill_tabs[0]:
                    # Create two columns for the overview
                    overview_col1, overview_col2 = st.columns(2)
                    
                    with overview_col1:
                        # Skills match ratio
                        match_ratio = len(skills_matched) / (len(job_skills) or 1)
                        st.metric(
                            "Skills Match Ratio",
                            f"{match_ratio:.0%}",
                            f"{len(skills_matched)}/{len(job_skills)} skills"
                        )
                    
                    with overview_col2:
                        # Critical skills status
                        critical_skills = set(job_skills) & {"Python", "Java", "JavaScript", "SQL", "AWS", "Docker"}  # Example critical skills
                        critical_matched = critical_skills & set(skills_matched)
                        st.metric(
                            "Critical Skills",
                            f"{len(critical_matched)}/{len(critical_skills)}",
                            "core skills matched" if critical_matched else "core skills needed"
                        )
                
                with skill_tabs[1]:
                    if skills_matched:
                        st.markdown("""
                            <div style='display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px;'>
                        """, unsafe_allow_html=True)
                        
                        for skill in sorted(skills_matched):
                            st.markdown(f"""
                                <span class='skill-tag skill-match' 
                                      style='font-size: 12px; padding: 2px 8px;'>
                                    ✓ {skill}
                                </span>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No matching skills found.")
                
                with skill_tabs[2]:
                    if skills_missing:
                        # Group missing skills by category
                        categories = {
                            "Programming": {"Python", "Java", "JavaScript", "C++", "Ruby"},
                            "Web": {"React", "Angular", "Vue.js", "HTML", "CSS"},
                            "Cloud": {"AWS", "Azure", "Docker", "Kubernetes"},
                            "Data": {"SQL", "MongoDB", "PostgreSQL", "MySQL"},
                            "Tools": {"Git", "JIRA", "Jenkins"}
                        }
                        
                        missing_by_category = {}
                        for skill in skills_missing:
                            categorized = False
                            for cat, cat_skills in categories.items():
                                if skill in cat_skills:
                                    missing_by_category.setdefault(cat, []).append(skill)
                                    categorized = True
                                    break
                            if not categorized:
                                missing_by_category.setdefault("Other", []).append(skill)
                        
                        # Display missing skills by category in columns
                        cols = st.columns(2)
                        for i, (category, skills) in enumerate(missing_by_category.items()):
                            with cols[i % 2]:
                                st.markdown(f"**{category}**")
                                for skill in sorted(skills):
                                    st.markdown(f"""
                                        <span class='skill-tag skill-missing' 
                                              style='font-size: 12px; padding: 2px 8px; margin: 2px;'>
                                            + {skill}
                                        </span>
                                    """, unsafe_allow_html=True)
                    else:
                        st.success("You have all the required skills!")
            
            with col2:
                st.markdown("### 📋 Job Details")
                
                # Create a styled info box
                st.markdown("""
                    <style>
                    .job-info {
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 5px;
                    }
                    </style>
                    <div class="job-info">
                """, unsafe_allow_html=True)
                
                st.markdown(f"🏢 **Company:** {job['company_name']}")
                st.markdown(f"📍 **Location:** {job['job_location']}")
                st.markdown(f"💼 **Type:** {job['job_employment_type']}")
                st.markdown(f"📊 **Level:** {job['job_seniority_level']}")
                if pd.notna(job['job_base_pay_range']):
                    st.markdown(f"💰 **Pay Range:** {job['job_base_pay_range']}")
                if pd.notna(job['job_posted_date']):
                    st.markdown(f"📅 **Posted:** {job['job_posted_date']}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add apply button with styling
                if pd.notna(job['apply_link']):
                    st.markdown("""
                        <style>
                        .apply-button {
                            background-color: #00cc00;
                            color: white;
                            padding: 10px 20px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            border-radius: 5px;
                            margin-top: 10px;
                            width: 100%;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    st.markdown(f"<a href='{job['apply_link']}' class='apply-button'>Apply Now</a>", unsafe_allow_html=True)

    # Match score visualization
    st.subheader("📊 Match Score Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_matches['job_title'] + " @ " + top_matches['company_name'], top_matches['match_score'], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel("Match Score")
    ax.set_title("Top 10 Job Matches")
    st.pyplot(fig)

    # Download CSV
    st.subheader("📥 Download Top Matches")
    csv = top_matches.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="top_job_matches.csv", mime="text/csv")
else:
    st.info("Please upload your resume to see matching jobs.")
