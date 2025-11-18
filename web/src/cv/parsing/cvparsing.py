"""
cvparsing.py

Core parsing logic for resumes.

Main entrypoint used by the web API:
    parse_single_resume(file_path: str) -> dict
"""

import os
import re
import json
import uuid
from typing import List, Dict

import fitz          # PyMuPDF
import spacy
import docx2txt


# ---------------------------
# SpaCy model (load once)
# ---------------------------

# Make sure you've run:
#   python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# ---------------------------
# Text extraction helpers
# ---------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file on disk."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file on disk."""
    return docx2txt.process(file_path)


# ---------------------------
# Field extraction helpers
# ---------------------------

def _extract_name_and_location(doc) -> Dict[str, str]:
    """Extract first PERSON as name and first GPE/LOC as location."""
    name = ""
    location = ""

    for ent in doc.ents:
        if not name and ent.label_ == "PERSON":
            name = ent.text
        if not location and ent.label_ in ("GPE", "LOC"):
            location = ent.text
        if name and location:
            break

    return {
        "name": name,
        "location": location
    }


def _extract_email_and_phone(text: str) -> Dict[str, str]:
    """Extract first email and phone using regex."""
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    phone_match = re.search(r"\+?\d[\d\s\-]{8,}\d", text)

    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""

    return {
        "email": email,
        "phone": phone
    }


def _extract_education(text: str) -> str:
    """
    Very simple education extraction:
    looks for common degree keywords.
    """
    pattern = r"(Bachelor|Bachelors|Master|Masters|B\.Sc|M\.Sc|Ph\.D|MBA)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return ""
    # Deduplicate while preserving order
    seen = set()
    cleaned = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(m)
    return ", ".join(cleaned)


def _extract_years_experience(text: str) -> str:
    """
    Look for patterns like '3 years', '5+ years' and return a string.
    """
    exp_match = re.search(r"(\d+)\+?\s+years", text, flags=re.IGNORECASE)
    if exp_match:
        return f"{exp_match.group(1)} years"
    return ""


def _extract_skills(text: str) -> List[str]:
    """
    Simple keyword-based skill extraction.
    You can expand/customize this list as you like.
    """
    SKILL_KEYWORDS = {
    "python": ["python", "python3", "cpython", "pypy"],
    "java": ["java", "openjdk", "jdk", "jvm"],
    "javascript": ["javascript", "js", "nodejs", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "go": ["go", "golang"],
    "c": ["c"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "ruby": ["ruby", "ruby on rails", "rails"],
    "php": ["php", "laravel", "symfony"],
    "rust": ["rust"],
    "kotlin": ["kotlin"],
    "scala": ["scala"],
    "r": ["r", "r language"],
    "swift": ["swift"],
    "shell": ["shell", "bash", "zsh", "sh"],
    "perl": ["perl"],
    "sql": ["sql", "mysql", "postgresql", "oracle", "sqlite", "mssql"],
    "nosql": ["nosql", "mongodb", "cassandra", "redis", "dynamodb", "couchdb"],
    "big data": ["big data", "hadoop", "spark", "mapreduce", "hive", "pig"],
    "data engineering": ["data engineering", "etl", "data pipeline", "airflow"],
    "data science": ["data science", "machine learning", "ml", "statistics"],
    "deep learning": ["deep learning", "neural networks", "tensorflow", "pytorch", "keras"],
    "mlops": ["mlops", "model deployment", "model serving", "sagemaker", "mlflow"],
    "nlp": ["nlp", "natural language processing", "transformers", "spacy", "nltk"],
    "computer vision": ["computer vision", "cv", "opencv"],
    "analytics": ["analytics", "business intelligence", "bi", "tableau", "power bi"],
    "rest": ["rest", "restful api", "api", "web api"],
    "graphql": ["graphql"],
    "web frameworks": ["django", "flask", "express", "spring", "rails", "fastapi"],
    "frontend frameworks": ["react", "angular", "vue", "svelte", "ember"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "scss", "sass", "less"],
    "webpack": ["webpack", "rollup", "parcel"],
    "microservices": ["microservices", "service oriented architecture", "soa"],
    "devops": ["devops", "ci/cd", "continuous integration", "continuous delivery", "continuous deployment"],
    "docker": ["docker", "containers"],
    "kubernetes": ["kubernetes", "k8s", "kube"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "chef": ["chef"],
    "puppet": ["puppet"],
    "helm": ["helm"],
    "istio": ["istio", "service mesh"],
    "prometheus": ["prometheus", "grafana", "monitoring"],
    "logging": ["elk", "elasticsearch", "logstash", "kibana", "splunk"],
    "cloud aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation", "iam", "dynamodb"],
    "cloud azure": ["azure", "microsoft azure", "azure functions", "azure devops", "arm templates"],
    "cloud gcp": ["gcp", "google cloud", "google cloud platform", "gce", "bigquery", "cloud functions"],
    "openstack": ["openstack"],
    "serverless": ["serverless", "faas"],
    "edge computing": ["edge computing"],
    "networking": ["networking", "dns", "http", "tcp/ip"],
    "security": ["security", "kubernetes security", "oauth2", "jwt", "tls", "ssl", "vault"],
    "git": ["git", "gitlab", "github", "bitbucket"],
    "ci tools": ["jenkins", "circleci", "travis ci", "github actions", "gitlab ci", "azure pipelines"],
    "jira": ["jira", "confluence"],
    "slack": ["slack"],
    "docker-compose": ["docker-compose", "compose"],
    "testing": ["testing", "unit test", "integration test", "pytest", "junit", "mocha", "jest"],
    "performance": ["performance", "profiling", "benchmark"],
    "cache": ["cache", "redis", "memcached"],
    "microservices architecture": ["microservices architecture", "soa", "service mesh"],
    "design patterns": ["design patterns", "solid", "ddd", "clean architecture"],
    "architecture": ["architecture", "system design", "scalability", "high availability"],
    "agile": ["agile", "scrum", "kanban"],
    "devsecops": ["devsecops", "security as code", "shift left"],
    "observability": ["observability", "opentelemetry", "logging", "tracing", "metrics"]
    }
    found = []
    lowered = text.lower()
    for kw in SKILL_KEYWORDS:
        if kw in lowered:
            found.append(kw.title())
    # dedupe
    return sorted(set(found))


def _extract_certifications(text: str) -> List[str]:
    """
    Basic certification detection based on keywords.
    """
    pattern = r"(certificate|certified|certification|AWS|Azure|Google Cloud|Snowflake|PMP)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return []
    # dedupe with capitalization normalized
    return sorted(set(m.strip() for m in matches))


def extract_basic_fields(text: str) -> Dict:
    """
    Run spaCy + regex helpers and produce a clean dict of parsed fields.
    This is what your frontend will display.
    """
    doc = nlp(text)

    base = {}
    base["id"] = str(uuid.uuid4())

    # Name + location
    nl = _extract_name_and_location(doc)
    base["name"] = nl["name"] or "Unknown"
    base["location"] = nl["location"] or ""

    # Email + phone
    ep = _extract_email_and_phone(text)
    base["email"] = ep["email"]
    base["phone"] = ep["phone"]

    # Education / experience / skills / certs
    base["education"] = _extract_education(text)
    base["years_experience"] = _extract_years_experience(text)
    base["skills"] = _extract_skills(text)
    base["certifications"] = _extract_certifications(text)

    # You can expand these later if you start extracting them
    base["current_title"] = ""
    base["industries"] = []
    base["achievements"] = []

    return base


# ---------------------------
# Public API used by webapp
# ---------------------------

def parse_single_resume(file_path: str) -> Dict:
    """
    Parse a single resume file (PDF or DOCX) and return a dict
    with fields like:
      {
        "id": ...,
        "name": ...,
        "email": ...,
        "phone": ...,
        "education": ...,
        "years_experience": ...,
        "skills": [...],
        "certifications": [...],
        "location": ...,
        "current_title": "",
        "industries": [],
        "achievements": []
      }
    """
    
    ext = os.path.splitext(file_path)[1].lower()

    # 1) Get the raw text from the file
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # 2) Run your existing text-to-fields logic
    #    Change 'extract_basic_fields' to whatever your function is called.
    parsed = extract_basic_fields(text)  # or extract_entities_from_text(text)

    return parsed
    # ext = os.path.splitext(file_path)[1].lower()

    # if ext == ".pdf":
    #     text = extract_text_from_pdf(file_path)
    # elif ext in (".docx", ".doc"):
    #     text = extract_text_from_docx(file_path)
    # else:
    #     raise ValueError(f"Unsupported file type: {ext}")

    # return extract_basic_fields(text)


# ---------------------------
# Optional: batch helper
# ---------------------------

def parse_resumes_from_folder(folder_path: str, output_file: str) -> None:
    """
    Optional utility if you ever want to batch-parse resumes.

    NOTE: this DOES NOT hard-code any folder paths; you pass the
    folder and output file explicitly when you call it.
    """
    all_results: List[Dict] = []

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".pdf", ".docx", ".doc"):
            continue

        try:
            parsed = parse_single_resume(fpath)
            parsed["source_file"] = fname
            all_results.append(parsed)
        except Exception as e:
            print(f"Error parsing {fname}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Parsed {len(all_results)} resumes to {output_file}")


# ---------------------------
# CLI entry (manual use)
# ---------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        # Parse a single file: python cvparsing.py path/to/resume.pdf
        path = sys.argv[1]
        result = parse_single_resume(path)
        print(json.dumps(result, indent=2))
    elif len(sys.argv) == 3:
        # Batch mode: python cvparsing.py folder_path output.json
        folder = sys.argv[1]
        out = sys.argv[2]
        parse_resumes_from_folder(folder, out)
    else:
        print("Usage:")
        print("  python cvparsing.py resume.pdf")
        print("  python cvparsing.py folder_path output.json")
