import re
import json
import pdfplumber
import docx2txt
import spacy
import pandas as pd
import os
from pathlib import Path

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# 📄 Extract text from PDF or DOCX
def extract_text(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif ext in [".docx", ".doc"]:
        return docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file format")

# 📧 Email extraction
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None

# 📆 Experience extraction
def extract_experience(text):
    matches = re.findall(r"(\d+)\s*\+?\s*(?:years?|yrs?)\b", text, re.IGNORECASE)
    return f"{max(map(int, matches))} years" if matches else "Not specified"

# 🛠 Skill extraction
def extract_skills(text, keywords):
    found = []
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b' if ' ' not in kw else re.escape(kw)
        if re.search(pattern, text, re.IGNORECASE):
            found.append(kw)
    return sorted(set(found))

# 🧪 Project extraction
def extract_projects(text):
    projects = re.findall(r"(?:Project[s]?:|Projects Worked On:)(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    return [proj.strip() for proj in projects] if projects else []

# 🏆 Achievements extraction
def extract_achievements(text):
    achievements = re.findall(r"(?:Achievements|Accomplishments|Awards)(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    return [a.strip() for a in achievements] if achievements else []

# 🏢 Company/Industry extraction
def extract_companies(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))

# 🧠 Main parser
def parse_resume(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    text = extract_text(file_path)
    if not text or not text.strip():
        raise ValueError(f"No extractable text found in {file_path}. Try using OCR (e.g., pytesseract).")

    technical_keywords = ["Python", "Java", "SQL", "AWS", "Docker", "Kubernetes", "TensorFlow", "React", "Node.js"]
    managerial_keywords = ["Agile", "Scrum", "Project Management", "Team Lead", "Stakeholder", "Budgeting"]

    data = {
        "email_id": extract_email(text),
        "total_experience": extract_experience(text),
        "technical_skillset": extract_skills(text, technical_keywords),
        "managerial_skillset": extract_skills(text, managerial_keywords),
        "achievements": extract_achievements(text),
        "projects_mentioned": extract_projects(text),
        "companies_or_industries": extract_companies(text)
    }

    return data

# 📤 Save to JSON and CSV
def save_outputs(data, json_path="resume_output.json", csv_path="resume_output.csv"):
    with open(json_path, "w") as jf:
        json.dump(data, jf, indent=4)

    flat_data = {k: ", ".join(v) if isinstance(v, list) else v for k, v in data.items()}
    df = pd.DataFrame([flat_data])
    df.to_csv(csv_path, index=False)

# 🚀 Example usage (set absolute path)
if __name__ == "__main__":
    # 👇 Change only the filename here if your resume has a different name
    file_path = r"C:\Users\takbh\BDA696\venv2\how-I-met-my-job\Resume\AyushiSharma_resume.pdf"
    
    print(f"🔍 Looking for file: {file_path}")
    
    parsed_data = parse_resume(file_path)
    save_outputs(parsed_data)
    
    print("\n✅ Resume parsed successfully!\n")
    print(json.dumps(parsed_data, indent=4))
