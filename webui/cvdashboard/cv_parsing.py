import os
import fitz  # PyMuPDF
import re
import json
import uuid
import tempfile
import docx2txt

# Lazy spaCy loader to avoid importing heavy C-extensions at module import time
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
    return _nlp


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes (for uploaded files)."""
    text = ""
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        for page in pdf:
            text += page.get_text()
        pdf.close()
    except Exception:
        text = ""
    return text


def extract_text_from_docx_bytes(data: bytes) -> str:
    """Extract text from DOCX bytes by writing to a temp file then using docx2txt."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        text = docx2txt.process(tmp_path)
    except Exception:
        text = ""
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return text


def extract_entities(text: str) -> dict:
    """Extract structured information from resume text.

    If spaCy is available, use its NER and POS tagging. Otherwise fall back
    to regex-based extraction so the API can run without spaCy installed.
    """
    nlp = get_nlp()

    name = ""
    location = ""
    current_title = ""

    if nlp:
        doc = nlp(text)
        name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "")
        location = next((ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]), "")
        for token in doc:
            if token.pos_ == "PROPN" and token.text != name:
                current_title = token.text
                break

    # Regex-based details (always run)
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d\s\-]{8,}\d", text)
    education = re.findall(r"(Bachelor|Master|B\.Sc|M\.Sc|Ph\.D|MBA)", text, re.I)
    skills = re.findall(r"(Python|SQL|Excel|Machine Learning|Java|AWS|Power BI|Tableau|Hadoop|Spark|R|Snowflake|TensorFlow|Data Analysis)", text, re.I)
    certifications = re.findall(r"(Certified|Certificate|Certification|AWS|Azure|Google Cloud|Snowflake|PMP)", text, re.I)
    achievements = re.findall(r"(award|achievement|recognition|promotion|success|honor)", text, re.I)

    exp_match = re.search(r"(\d+)\+?\s+years", text)
    years_experience = float(exp_match.group(1)) if exp_match else 0.0

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


def parse_file_bytes(filename: str, data: bytes) -> dict:
    """Parse a single resume provided as bytes and return structured JSON-like dict."""
    lower = filename.lower()
    text = ""
    if lower.endswith('.pdf'):
        text = extract_text_from_pdf_bytes(data)
    elif lower.endswith('.docx'):
        text = extract_text_from_docx_bytes(data)
    else:
        # try to treat as text
        try:
            text = data.decode('utf-8', errors='ignore')
        except Exception:
            text = ''

    entities = extract_entities(text)

    # Build a more frontend-friendly shape
    result = {
        'name': entities.get('full_name') or entities.get('Name') or 'Unknown',
        'email': re.search(r"[\w\.-]+@[\w\.-]+", text).group(0) if re.search(r"[\w\.-]+@[\w\.-]+", text) else '',
        'phone': re.search(r"\+?\d[\d\s\-]{8,}\d", text).group(0) if re.search(r"\+?\d[\d\s\-]{8,}\d", text) else '',
        'education': entities.get('education_level') or '',
        'experience': f"{entities.get('years_experience', 0)} years",
        'skills': entities.get('skills') or [],
        'certifications': entities.get('certifications') or [],
        'raw_text': text
    }

    return result


if __name__ == "__main__":
    folder_path = r"C:\Users\takbh\OneDrive\Desktop\BDA696\project\how-I-met-my-job\dataset\Resume"
    output_file = os.path.join(os.path.dirname(__file__), "parsed_candidates.json")
    parse_resumes_from_folder(folder_path, output_file)
