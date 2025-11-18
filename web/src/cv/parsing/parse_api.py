# parse_api.py
import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

from cvparsing import parse_single_resume  # same folder import

app = Flask(__name__)
CORS(app)

@app.post("/parse-resume")
def parse_resume_endpoint():
    """
    Accept a resume upload (PDF), parse it with cvparsing.parse_single_resume,
    and return JSON to the browser.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(upload.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload.save(tmp.name)
        tmp_path = tmp.name

    try:
        parsed = parse_single_resume(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Failed to parse resume: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return jsonify(parsed)


if __name__ == "__main__":
    # Run on http://127.0.0.1:5001
    app.run(host="127.0.0.1", port=5001, debug=True)
