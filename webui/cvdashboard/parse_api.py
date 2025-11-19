from flask import Flask, request, jsonify, send_from_directory
import os
from cv_parsing import parse_file_bytes

app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/api/parse', methods=['POST'])
def parse_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part named resume'}), 400

    f = request.files['resume']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    data = f.read()
    result = parse_file_bytes(f.filename, data)

    # Add permissive CORS header for local dev (you can tighten later)
    resp = jsonify(result)
    resp.headers.add('Access-Control-Allow-Origin', '*')
    return resp


@app.route('/')
def index():
    # Serve the index.html in this folder
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
