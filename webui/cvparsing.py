import os
import fitz  # PyMuPDF
import re
import json
import uuid
import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_entities(text: str) -> dict:
    """Extract structured information from resume text."""
    doc = nlp(text)

    # Named entities
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "")
    location = next((ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]), "")

    # Regex-based details
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d\s\-]{8,}\d", text)
    education = re.findall(r"(Bachelor|Master|B\.Sc|M\.Sc|Ph\.D|MBA)", text, re.I)
    skills = re.findall(r"(Python|SQL|Excel|Machine Learning|Java|AWS|Power BI|Tableau|Hadoop|Spark|R|Snowflake|TensorFlow|Data Analysis)", text, re.I)
    certifications = re.findall(r"(Certified|Certificate|Certification|AWS|Azure|Google Cloud|Snowflake|PMP)", text, re.I)
    achievements = re.findall(r"(award|achievement|recognition|promotion|success|honor)", text, re.I)

    exp_match = re.search(r"(\d+)\+?\s+years", text)
    years_experience = float(exp_match.group(1)) if exp_match else 0.0

    # Guess current title
    current_title = ""
    for token in doc:
        if token.pos_ == "PROPN" and token.text != name:
            current_title = token.text
            break

    return {
        "Name": name or "Unknown",
        "candidate_id": str(uuid.uuid4()),
        "full_name": name or "",
        "location": location or "",
        "education_level": education[0] if education else "",
        "years_experience": years_experience,
        "skills": list(set(skills)),
        "certifications": list(set(certifications)),
        "current_title": current_title,
        "industries": [],
        "achievements": list(set(achievements))
    }

def parse_resumes_from_folder(folder_path: str, output_file: str):
    """Parse all PDF resumes in a folder and save results to one JSON file."""
    all_candidates = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"📄 Processing: {file_name}")

            try:
                text = extract_text_from_pdf(file_path)
                candidate_data = extract_entities(text)
                all_candidates.append(candidate_data)
            except Exception as e:
                print(f"⚠️ Error reading {file_name}: {e}")

    # Save all results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=4)

    print(f"\n✅ Extracted data for {len(all_candidates)} resumes saved to: {output_file}")


# ----------- RUN SCRIPT -----------
if __name__ == "__main__":
    folder_path = r"C:\Users\takbh\OneDrive\Desktop\BDA696\project\how-I-met-my-job\dataset\Resume"
    output_file = "parsed_candidates.json"

    parse_resumes_from_folder(folder_path, output_file)
