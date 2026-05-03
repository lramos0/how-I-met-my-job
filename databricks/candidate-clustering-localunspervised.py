"""
K-Means Clustering for Candidate Segmentation
Local version using scikit-learn - Uses real resume data
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# PDF and DOCX parsing
try:
    import PyPDF2
    import docx
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'PyPDF2', 'python-docx'])
    import PyPDF2
    import docx

print("=" * 80)
print("CANDIDATE CLUSTERING - REAL RESUME DATA")
print("=" * 80)

# ============================================================================
# 1. PARSE REAL RESUMES FROM DATASET
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text
    except Exception as e:
        print(f"  ⚠ Error reading {pdf_path.name}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"  ⚠ Error reading {docx_path.name}: {e}")
        return ""

def extract_skills(text):
    """Extract technical skills from resume text using comprehensive skill dictionary"""
    # Comprehensive skill dictionary (from skill_dict.js)
    SKILL_DICT = {
        "Python": ["python", "python3", "cpython", "pypy"],
        "Java": ["java", "openjdk", "jdk", "jvm"],
        "JavaScript": ["javascript", "js", "nodejs", "ecmascript"],
        "TypeScript": ["typescript", "ts"],
        "C": [r"\bc\b", " c programming", "c language"],
        "C++": ["c++", "cpp"],
        "C#": ["c#", "csharp"],
        "R": [r"\br\b", "r language", "r script"],
        "SQL": [r"\bsql\b", "mysql", "postgresql", "postgres", "oracle", "sqlite", "mssql", "sql server"],
        "NoSQL": ["nosql", "mongodb", "cassandra", "redis", "dynamodb", "couchdb"],
        "Tableau": ["tableau", "tableau desktop", "tableau server"],
        "Power BI": ["power bi", "powerbi"],
        "MicroStrategy": ["microstrategy"],
        "Excel": ["excel", "microsoft excel", "excel vba", "vba"],
        "Spark": ["spark", "apache spark"],
        "Hadoop": ["hadoop", "mapreduce", "hive"],
        "Big Data": ["big data", "bigdata"],
        "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch"],
        "Keras": ["keras"],
        "Scikit-Learn": ["scikit-learn", "sklearn", "scikit"],
        "XGBoost": ["xgboost"],
        "LightGBM": ["lightgbm"],
        "Machine Learning": ["machine learning", "ml"],
        "Deep Learning": ["deep learning", "neural network", "neural networks"],
        "Data Science": ["data science", "data scientist", "data analytics", "analytics"],
        "Data Engineering": ["data engineering", "etl", "data pipeline", "airflow", "dbt"],
        "Airflow": ["airflow"],
        "MLOps": ["mlops", "model deployment", "model serving", "sagemaker", "mlflow"],
        "NLP": ["nlp", "natural language processing", "transformers", "spacy", "nltk"],
        "Computer Vision": ["computer vision", "opencv", "cv"],
        "Docker": ["docker", "containers"],
        "Docker Compose": ["docker-compose", "compose"],
        "Kubernetes": ["kubernetes", "k8s", "kube"],
        "Helm": ["helm"],
        "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation", "iam", "dynamodb"],
        "Azure": ["azure", "microsoft azure", "azure functions", "azure devops"],
        "GCP": ["gcp", "google cloud", "google cloud platform", "bigquery", "gce"],
        "DevOps": ["devops", "ci/cd", "continuous integration", "continuous delivery", "continuous deployment"],
        "CI/CD": ["jenkins", "circleci", "travis", "github actions", "gitlab ci", "azure pipelines"],
        "Git": ["git", "github", "gitlab", "bitbucket"],
        "REST API": ["rest", "restful api", "api", "web api"],
        "GraphQL": ["graphql"],
        "Django": ["django"],
        "Flask": ["flask"],
        "FastAPI": ["fastapi"],
        "Node.js": ["node", "nodejs", "node.js"],
        "Express": ["express"],
        "Spring": ["spring", "spring boot"],
        "Hibernate": ["hibernate"],
        "SQLAlchemy": ["sqlalchemy"],
        "React": ["react", "reactjs"],
        "Angular": ["angular"],
        "Vue.js": ["vue", "vue.js"],
        "Svelte": ["svelte"],
        "React Native": ["react native", "react-native"],
        "Flutter": ["flutter", "dart"],
        "Android": ["android", "android sdk"],
        "iOS": ["ios", "ios sdk", "objective-c", "swift"],
        "Mobile Development": ["mobile", "android", "ios"],
        "UX Design": ["ux", "user experience", "user research", "wireframing", "prototyping", "figma", "adobe xd"],
        "Figma": ["figma"],
        "Sketch": ["sketch"],
        "Photoshop": ["photoshop"],
        "Illustrator": ["illustrator"],
        "Business Intelligence": ["business intelligence", "bi", "tableau", "power bi", "microstrategy", "looker", "qlik"],
        "Looker": ["looker"],
        "Qlik": ["qlik", "qlikview", "qliksense"],
        "DBT": ["dbt"],
        "Fivetran": ["fivetran"],
        "Stitch": ["stitch data", "stitch"],
        "Kafka": ["kafka", "apache kafka"],
        "RabbitMQ": ["rabbitmq"],
        "Redis": ["redis"],
        "Elasticsearch": ["elasticsearch", "elk", "elastic"],
        "Neo4j": ["neo4j", "graph database"],
        "MATLAB": ["matlab"],
        "SAS": ["sas"],
        "SPSS": ["spss"],
        "Julia": ["julia"],
        "Blockchain": ["blockchain", "ethereum", "smart contract", "solidity"],
        "Solidity": ["solidity"],
        "Unity": ["unity"],
        "Unreal Engine": ["unreal engine"],
        "Datadog": ["datadog"],
        "New Relic": ["new relic"],
        "Monitoring": ["prometheus", "grafana", "monitoring"],
        "Logging": ["elk", "elasticsearch", "logstash", "kibana", "splunk"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "Chef": ["chef"],
        "Puppet": ["puppet"],
        "Vagrant": ["vagrant"],
        "VMware": ["vmware"],
        "OpenShift": ["openshift"],
        "Security": ["security", "oauth2", "jwt", "tls", "ssl", "vault", "gdpr", "hipaa"],
        "Networking": ["networking", "dns", "tcp/ip", "http"],
        "Project Management": ["project management", "project manager", "pmp", "scrum", "agile", "kanban"],
        "Product Management": ["product management", "product manager", "roadmap", "user stories"],
        "Business Analysis": ["business analysis", "business analyst", "requirements gathering"],
        "Accessibility": ["a11y", "accessibility", "wcag"],
        "Data Visualization": ["data visualization", "d3", "d3.js", "matplotlib", "seaborn", "ggplot2"],
        "Matplotlib": ["matplotlib"],
        "Seaborn": ["seaborn"],
        "ggplot2": ["ggplot2"],
        "Testing": ["testing", "selenium", "cypress", "unit test", "integration test", "pytest", "junit", "mocha", "jest"],
        "Socket.io": ["socket.io"],
        "Electron": ["electron"],
        "Encryption": ["encryption", "openssl"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "LookML": ["lookml"],
        "Tableau Prep": ["tableau prep"],
        "Power Query": ["power query"]
    }
    
    found_skills = []
    text_lower = text.lower()
    
    # Check each skill and its aliases
    for skill_name, aliases in SKILL_DICT.items():
        for alias in aliases:
            # Use regex for word boundary patterns, simple search for others
            if alias.startswith(r'\b'):
                if re.search(alias, text_lower):
                    found_skills.append(skill_name)
                    break
            else:
                if alias in text_lower:
                    found_skills.append(skill_name)
                    break
    
    return found_skills

def extract_education(text):
    """Extract education level from resume"""
    text_lower = text.lower()
    if 'ph.d' in text_lower or 'phd' in text_lower or 'doctorate' in text_lower:
        return 'PhD'
    elif 'master' in text_lower or 'm.s.' in text_lower or 'msc' in text_lower or 'm.tech' in text_lower:
        return 'Master'
    elif 'bachelor' in text_lower or 'b.s.' in text_lower or 'b.tech' in text_lower or 'b.e.' in text_lower:
        return 'Bachelor'
    elif 'associate' in text_lower:
        return 'Associate'
    else:
        return 'Bachelor'  # Default

def extract_years_experience(text):
    """Estimate years of experience from resume"""
    # Look for years mentioned (e.g., "5 years", "3+ years")
    years_pattern = r'(\d+)[\s]*[\+]*[\s]*(years?|yrs?)'
    matches = re.findall(years_pattern, text.lower())
    if matches:
        years = [int(m[0]) for m in matches]
        return max(years) if years else 0
    
    # Try to extract from date ranges (e.g., 2020-2023)
    date_pattern = r'(20\d{2})\s*[-–—]\s*(20\d{2}|present|current)'
    date_matches = re.findall(date_pattern, text.lower())
    if date_matches:
        total_years = 0
        for start, end in date_matches:
            end_year = 2025 if end in ['present', 'current'] else int(end)
            total_years += end_year - int(start)
        return min(total_years, 30)  # Cap at 30 years
    
    return np.random.randint(1, 8)  # Random fallback

def extract_title(text):
    """Extract job title from resume"""
    titles = ['Data Engineer', 'Data Scientist', 'Software Engineer', 'Data Analyst', 
              'Machine Learning Engineer', 'ML Engineer', 'Developer', 'Analyst',
              'Engineer', 'Consultant', 'Manager', 'Architect', 'Scientist']
    
    text_lower = text.lower()
    for title in titles:
        if title.lower() in text_lower:
            return title
    return 'Software Engineer'  # Default

def extract_location(text):
    """Extract location from resume"""
    us_cities = [
        'San Francisco', 'Seattle', 'New York', 'Austin', 'Boston',
        'Chicago', 'Los Angeles', 'Denver', 'Atlanta', 'Dallas',
        'San Diego', 'Portland', 'Miami', 'Phoenix', 'Washington'
    ]
    
    for city in us_cities:
        if city.lower() in text.lower():
            return city + ', USA'
    
    # Check for Indian cities
    indian_cities = ['Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Pune', 'Chennai']
    for city in indian_cities:
        if city.lower() in text.lower():
            return city + ', India'
    
    return 'Unknown'

# Parse all resumes
resume_dir = Path('dataset/Resume')
data = []

print("\n📄 Parsing resumes from:", resume_dir.absolute())
print("=" * 80)

for resume_file in resume_dir.iterdir():
    if resume_file.suffix.lower() in ['.pdf', '.docx']:
        print(f"  Processing: {resume_file.name}")
        
        # Extract text
        if resume_file.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(resume_file)
        else:
            text = extract_text_from_docx(resume_file)
        
        if not text.strip():
            print(f"  ⚠ Skipping (empty): {resume_file.name}")
            continue
        
        # Extract features
        candidate = {
            'candidate_id': f'CAND-{len(data)+1:03d}',
            'full_name': resume_file.stem.replace('_', ' ').replace('-', ' '),
            'location': extract_location(text),
            'education_level': extract_education(text),
            'years_experience': extract_years_experience(text),
            'skills': ', '.join(extract_skills(text)),
            'current_title': extract_title(text),
            'resume_file': resume_file.name,
            'text_length': len(text)
        }
        data.append(candidate)
        print(f"    ✓ Extracted: {candidate['current_title']}, {candidate['education_level']}, {len(extract_skills(text))} skills")

df = pd.DataFrame(data)
print(f"\n✓ Successfully parsed {len(df)} resumes")
print(f"✓ Total unique skills found: {len(set([s for skills in df['skills'] for s in skills.split(', ') if s]))}")
print("\n" + df[['candidate_id', 'full_name', 'current_title', 'education_level', 'years_experience']].to_string())
print("\n" + "=" * 80)

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# One-Hot Encode categorical features
ohe_features = ['education_level', 'current_title', 'location']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_encoded = ohe.fit_transform(df[ohe_features])
ohe_feature_names = ohe.get_feature_names_out(ohe_features)

print(f"✓ One-Hot Encoded features: {len(ohe_feature_names)} dimensions")

# Vectorize skills (CountVectorizer for text)
cv = CountVectorizer(max_features=50, min_df=2)
skills_encoded = cv.fit_transform(df['skills']).toarray()

print(f"✓ Skills vectorized: {skills_encoded.shape[1]} dimensions")

# Combine all features
X = np.hstack([
    ohe_encoded,
    skills_encoded,
    df[['years_experience']].values
])

print(f"✓ Total features: {X.shape[1]} dimensions")

# Scale features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Features scaled with StandardScaler")

# ============================================================================
# 3. DETERMINE OPTIMAL K (Elbow Method + Silhouette)
# ============================================================================

print("\n" + "=" * 80)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 80)

k_range = range(3, 10)
silhouette_scores = []
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_
    
    silhouette_scores.append(silhouette)
    inertias.append(inertia)
    
    print(f"k={k} | Silhouette: {silhouette:.4f} | Inertia: {inertia:.2f}")

# Select optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
optimal_silhouette = max(silhouette_scores)

print(f"\n✓ OPTIMAL CLUSTERS: k={optimal_k} (Silhouette: {optimal_silhouette:.4f})")

# ============================================================================
# 4. TRAIN FINAL MODEL
# ============================================================================

print("\n" + "=" * 80)
print(f"TRAINING FINAL K-MEANS MODEL (k={optimal_k})")
print("=" * 80)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
df['cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"✓ Model trained successfully")
print(f"  • Number of clusters: {optimal_k}")
print(f"  • Silhouette Score: {optimal_silhouette:.4f}")
print(f"  • Inertia: {kmeans_final.inertia_:.2f}")

# ============================================================================
# 5. CLUSTER ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER STATISTICS")
print("=" * 80)

cluster_stats = df.groupby('cluster').agg({
    'candidate_id': 'count',
    'years_experience': ['mean', 'std'],
}).round(2)

cluster_stats.columns = ['count', 'avg_years_exp', 'std_years_exp']
print(cluster_stats)

# ============================================================================
# 6. DETAILED CLUSTER PROFILES
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER PROFILES")
print("=" * 80)

for cluster_id in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster_id]
    cluster_size = len(cluster_data)
    
    print(f"\n{'=' * 80}")
    print(f"CLUSTER {cluster_id} - {cluster_size} Candidates ({cluster_size/len(df)*100:.1f}%)")
    print('=' * 80)
    
    print(f"\n📊 Statistics:")
    print(f"  • Avg Experience: {cluster_data['years_experience'].mean():.1f} years")
    print(f"  • Experience Range: {cluster_data['years_experience'].min()}-{cluster_data['years_experience'].max()} years")
    
    print(f"\n💼 Top Job Titles:")
    top_titles = cluster_data['current_title'].value_counts().head(3)
    for i, (title, count) in enumerate(top_titles.items(), 1):
        pct = (count / cluster_size) * 100
        print(f"  {i}. {title}: {count} ({pct:.1f}%)")
    
    print(f"\n🎓 Education Distribution:")
    edu_dist = cluster_data['education_level'].value_counts()
    for edu, count in edu_dist.items():
        pct = (count / cluster_size) * 100
        print(f"  • {edu}: {count} ({pct:.1f}%)")
    
    print(f"\n📍 Top Locations:")
    top_locs = cluster_data['location'].value_counts().head(3)
    for loc, count in top_locs.items():
        pct = (count / cluster_size) * 100
        print(f"  • {loc}: {count} ({pct:.1f}%)")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Create output directory
output_dir = Path('unsupervisedmloutput')
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory: {output_dir.absolute()}")

# Plot 1: Elbow Method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
plt.legend()
plt.grid(True)

# Plot 2: Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(output_dir / 'cluster_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: unsupervisedmloutput/cluster_evaluation.png")

# Plot 3: Cluster Distribution
plt.figure(figsize=(10, 6))
cluster_counts = df['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color='steelblue', edgecolor='black')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Candidates')
plt.title(f'Cluster Distribution (k={optimal_k})')
for i, v in enumerate(cluster_counts.values):
    plt.text(i, v + 1, str(v), ha='center', va='bottom')
plt.savefig(output_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: unsupervisedmloutput/cluster_distribution.png")

# Plot 4: Experience by Cluster
plt.figure(figsize=(10, 6))
df.boxplot(column='years_experience', by='cluster', figsize=(10, 6))
plt.xlabel('Cluster ID')
plt.ylabel('Years of Experience')
plt.title('Experience Distribution by Cluster')
plt.suptitle('')  # Remove default title
plt.savefig(output_dir / 'experience_by_cluster.png', dpi=300, bbox_inches='tight')
print("✓ Saved: unsupervisedmloutput/experience_by_cluster.png")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save clustered data
output_file = output_dir / 'candidate_clusters.csv'
df.to_csv(output_file, index=False)
print(f"✓ Saved clustered data to: {output_file}")

# Save cluster summary
summary_file = output_dir / 'cluster_summary.txt'
with open(summary_file, 'w') as f:
    f.write("CLUSTER SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Optimal number of clusters: {optimal_k}\n")
    f.write(f"Silhouette Score: {optimal_silhouette:.4f}\n")
    f.write(f"Total candidates: {len(df)}\n\n")
    f.write(cluster_stats.to_string())

print(f"✓ Saved summary to: {summary_file}")

print("\n" + "=" * 80)
print("CLUSTERING COMPLETE!")
print("=" * 80)
print(f"""
✓ K-Means clustering model trained with {optimal_k} clusters
✓ Silhouette Score: {optimal_silhouette:.4f}
✓ All outputs saved to: unsupervisedmloutput/
✓ Generated visualizations:
  - unsupervisedmloutput/cluster_evaluation.png
  - unsupervisedmloutput/cluster_distribution.png
  - unsupervisedmloutput/experience_by_cluster.png
✓ Results saved:
  - unsupervisedmloutput/candidate_clusters.csv
  - unsupervisedmloutput/cluster_summary.txt
""")
