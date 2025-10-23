# How I Met My JOB 🔍

An AI-powered job matching platform that helps job seekers find the most relevant positions by analyzing their resumes and comparing them with job postings using advanced natural language processing.

## 🌟 Features

- **Smart Resume Analysis**: Extract key information from PDF resumes
- **Intelligent Job Matching**: Using sentence transformers for semantic matching
- **Advanced Filtering**: Filter jobs by company, location, seniority, and more
- **Interactive UI**: User-friendly interface with visualizations
- **Export Functionality**: Download matched jobs as CSV

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lramos0/how-I-met-my-job.git
cd how-I-met-my-job
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running the Application

1. Navigate to the project directory:
```bash
cd webui
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to: http://localhost:8501

## 🔧 Project Structure

```
how-I-met-my-job/
├── webui/
│   ├── app.py                 # Main Streamlit application
│   ├── Streamlit_UI_integration.py  # UI components
│   ├── Streamlitfilters.py    # Filter functionality
│   └── data/
│       └── job_postings.csv   # Job listings dataset
├── .venv/                     # Virtual environment
└── README.md                  # This file
```

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **sentence-transformers**: Text similarity matching
- **PyMuPDF**: PDF processing
- **pandas**: Data manipulation
- **spaCy**: Natural language processing
- **matplotlib**: Data visualization

## 📊 Features in Detail

### Resume Processing
- PDF text extraction
- Skills identification
- Experience analysis

### Job Matching
- Semantic similarity scoring
- Weighted criteria matching
- Customizable filters

### User Interface
- Interactive filters
- Real-time updates
- Visual match scoring
- Export functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- [@lramos0](https://github.com/lramos0)

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by modern job search challenges
- Built with ❤️ for job seekers