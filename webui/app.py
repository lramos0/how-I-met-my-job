import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF
import docx2txt  # for DOCX
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import uuid
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="How I Met My JOB", layout="wide")

# =========================
# HELPER FUNCTIONS
# =========================
def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file uploaded via Streamlit."""
    text = ""
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        pdf_document.close()
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
    return text


def extract_text_from_docx(uploaded_file):
    """Extracts text from a DOCX file uploaded via Streamlit."""
    try:
        return docx2txt.process(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading DOCX: {e}")
        return ""


def compute_similarity(resume_text, df, model, skill_patterns):
    """Compute similarity score between resume and job descriptions."""
    # ✅ include job_summary since that's your text column
    possible_cols = ["job_summary", "job_description", "description", "summary", "details"]
    text_col = next((c for c in possible_cols if c in df.columns), None)
    if not text_col:
        st.error(f"❌ Could not find a column containing job descriptions. Found: {list(df.columns)}")
        st.stop()

    st.info(f"Using **{text_col}** as the job description column.")
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_embeddings = model.encode(df[text_col].astype(str).tolist(), convert_to_tensor=True)
    similarities = util.cos_sim(resume_emb, job_embeddings)[0].cpu().numpy()
    return similarities


# =========================
# LOAD MODELS AND DATA
# =========================
@st.cache_data
def load_job_data():
    path = r"C:\Users\takbh\OneDrive\Desktop\BDA696\project\how-I-met-my-job\webui\data\job_postings.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Optional: rename for consistency
    df.rename(columns={
        'company': 'company_name',
        'title': 'job_title',
        'location': 'job_location'
    }, inplace=True)

    return df


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


df = load_job_data()
model = load_model()
nlp = spacy.load("en_core_web_sm")

# Dummy skill patterns
SKILL_PATTERNS = ["Python", "SQL", "Machine Learning", "Data Analysis", "Tableau", "Excel"]

# =========================
# SESSION STATE
# =========================
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# =========================
# PAGE 1: Upload Resume
# =========================
if not st.session_state.resume_uploaded:
    st.title("📄 Upload Your Resume")
    st.markdown("Upload your resume (PDF or DOCX) to start personalized job matching!")

    uploaded_file = st.file_uploader("Choose your resume", type=["pdf", "docx"])

    if uploaded_file:
        with st.spinner("Reading and analyzing your resume..."):
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)

            if resume_text.strip():
                st.session_state.resume_uploaded = True
                st.session_state.resume_text = resume_text
                st.success("✅ Resume uploaded successfully!")
                st.rerun()
            else:
                st.error("Could not extract text from your file. Try another one.")
    st.stop()
# =========================
# PAGE 2: Job Matching + Resume Parsing + Filters
# =========================

st.title("🎯 Job Matching Dashboard")
st.markdown("Refine your job search and explore your best matches based on your parsed resume and job listings.")

# =========================================================
# 🧾 Resume Parsing Functionality
# =========================================================
def extract_entities_from_text(text: str) -> dict:
    """Extract structured info (skills, exp, certifications, achievements) from resume text."""
    doc = nlp(text)

    # Named entities
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "")
    location = next((ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]), "")

    # Regex-based extraction
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d\s\-]{8,}\d", text)
    education = re.findall(r"(Bachelor|Master|B\.Sc|M\.Sc|Ph\.D|MBA)", text, re.I)
    skills = re.findall(r"(Python|SQL|Excel|Machine Learning|Java|AWS|Power BI|Tableau|Hadoop|Spark|R|Snowflake|TensorFlow|Data Analysis)", text, re.I)
    certifications = re.findall(r"(Certified|Certificate|Certification|AWS|Azure|Google Cloud|Snowflake|PMP)", text, re.I)
    achievements = re.findall(r"(award|achievement|recognition|promotion|success|honor)", text, re.I)
    exp_match = re.search(r"(\d+)\+?\s+years", text)
    years_experience = float(exp_match.group(1)) if exp_match else 0.0

    return {
        "candidate_id": str(uuid.uuid4()),
        "full_name": name or "Unknown",
        "location": location or "",
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else "",
        "education_level": education[0] if education else "",
        "years_experience": years_experience,
        "skills": list(set(skills)),
        "certifications": list(set(certifications)),
        "achievements": list(set(achievements))
    }

# Run parser on uploaded resume text
resume_text = st.session_state.resume_text
parsed_data = extract_entities_from_text(resume_text)

# Save parsed JSON in project folder
output_dir = r"C:\Users\takbh\OneDrive\Desktop\BDA696\project\how-I-met-my-job\webui"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "parsed_resume.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(parsed_data, f, indent=4)

# =========================================================
# 🧾 Display Parsed Resume Summary
# =========================================================
st.markdown("---")
st.subheader("🧾 Resume Summary")

col1, col2 = st.columns(2)
with col1:
    ##st.markdown(f"**👤 Name:** {parsed_data['full_name']}")
    ##st.markdown(f"**📍 Location:** {parsed_data['location'] or 'N/A'}")
    st.markdown(f"**📧 Email:** {parsed_data['email'] or 'N/A'}")
    ##st.markdown(f"**📞 Phone:** {parsed_data['phone'] or 'N/A'}")
with col2:
    st.markdown(f"**🎓 Education:** {parsed_data['education_level'] or 'N/A'}")
    st.markdown(f"**💼 Experience:** {parsed_data['years_experience']} years")

st.markdown("### 🛠️ Skills")
if parsed_data["skills"]:
    st.success(", ".join(parsed_data["skills"]))
else:
    st.info("No skills detected.")

st.markdown("### 🏅 Certifications")
if parsed_data["certifications"]:
    st.success(", ".join(parsed_data["certifications"]))
else:
    st.info("No certifications detected.")

st.markdown("### 🌟 Achievements")
if parsed_data["achievements"]:
    st.success(", ".join(parsed_data["achievements"]))
else:
    st.info("No achievements detected.")

st.markdown("---")

# =========================================================
# 📋 Job Filters
# =========================================================
st.sidebar.header("📋 Filter Job Listings")

company = st.sidebar.selectbox("Company", ["All"] + sorted(df["company_name"].dropna().unique()), key="filter_company")
title = st.sidebar.selectbox("Job Title", ["All"] + sorted(df["job_title"].dropna().unique()), key="filter_title")
location = st.sidebar.selectbox("Job Location", ["All"] + sorted(df["job_location"].dropna().unique()), key="filter_location")
country = st.sidebar.selectbox("Country Code", ["All"] + sorted(df["country_code"].dropna().unique()), key="filter_country")
seniority = st.sidebar.selectbox("Seniority Level", ["All"] + sorted(df["job_seniority_level"].dropna().unique()), key="filter_seniority")
employment_type = st.sidebar.selectbox("Employment Type", ["All"] + sorted(df["job_employment_type"].dropna().unique()), key="filter_employment")
industry = st.sidebar.selectbox("Industry", ["All"] + sorted(df["job_industries"].dropna().unique()), key="filter_industry")
salary_range = st.sidebar.text_input("Base Pay Range (e.g. 50000-100000)", key="filter_salary")
date_range = st.sidebar.date_input("Posted Date Range", [], key="filter_daterange")

# Apply filters
filtered_df = df.copy()
if company != "All":
    filtered_df = filtered_df[filtered_df["company_name"] == company]
if title != "All":
    filtered_df = filtered_df[filtered_df["job_title"] == title]
if location != "All":
    filtered_df = filtered_df[filtered_df["job_location"] == location]
if country != "All":
    filtered_df = filtered_df[filtered_df["country_code"] == country]
if seniority != "All":
    filtered_df = filtered_df[filtered_df["job_seniority_level"] == seniority]
if employment_type != "All":
    filtered_df = filtered_df[filtered_df["job_employment_type"] == employment_type]
if industry != "All":
    filtered_df = filtered_df[filtered_df["job_industries"] == industry]

if salary_range:
    try:
        min_salary, max_salary = map(int, salary_range.split("-"))
        filtered_df = filtered_df[
            filtered_df["job_base_pay_range"].apply(
                lambda x: isinstance(x, str)
                and any(char.isdigit() for char in x)
                and min_salary <= int("".join(filter(str.isdigit, x))) <= max_salary
            )
        ]
    except:
        st.warning("Invalid salary range format. Use format like 50000-100000.")

if date_range:
    try:
        filtered_df["job_posted_date"] = pd.to_datetime(filtered_df["job_posted_date"], errors="coerce")
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df["job_posted_date"] >= pd.to_datetime(start_date))
                & (filtered_df["job_posted_date"] <= pd.to_datetime(end_date))
            ]
    except:
        st.warning("Invalid date format in job_posted_date column.")

# =========================================================
# 🔍 Compute Similarity with Resume
# =========================================================
st.subheader("📊 Job Matching Results")

with st.spinner("Analyzing job descriptions and computing similarity..."):
    # Ensure the right column is used
    if "job_summary" not in filtered_df.columns:
        st.error("❌ Could not find 'job_summary' column in job data.")
        st.stop()

    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_embeddings = model.encode(filtered_df["job_summary"].astype(str).tolist(), convert_to_tensor=True)
    similarities = util.cos_sim(resume_emb, job_embeddings)[0].cpu().numpy()
    filtered_df["match_score"] = similarities

# Sort by match score
top_matches = filtered_df.sort_values("match_score", ascending=False).head(10)

# Display top results
for _, job in top_matches.iterrows():
    st.markdown(
        f"**{job['job_title']}** at *{job['company_name']}* — "
        f"📍 {job.get('job_location', 'N/A')} — 🧠 Match Score: **{job['match_score']:.0%}**"
    )

# Visualization
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top_matches["job_title"], top_matches["match_score"])
ax.set_xlabel("Match Score")
ax.set_ylabel("Job Title")
ax.set_title("Top Matching Jobs")
st.pyplot(fig)

# Back button
if st.button("⬅️ Upload another resume"):
    st.session_state.resume_uploaded = False
    st.session_state.resume_text = ""
    st.rerun()
