# Databricks notebook source
# MAGIC %md
# MAGIC ## Generate Mock Data

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# pyspark==3.x
from pyspark.sql import Row, types as T
import random, json, datetime

random.seed(42)

# --- Vocabularies for job listings ---
job_titles = [
    "Software Engineer", "Data Scientist", "ML Engineer", "Product Manager", "UX Designer",
    "DevOps Engineer", "QA Engineer", "Business Analyst", "Cloud Architect", "Security Engineer"
]
companies = [
    "Techify", "DataWiz", "CloudNova", "InnoSoft", "FinEdge", "HealthSync", "RetailX", "AutoAI", "MediaWorks", "GreenEnergy"
]
locations = [
    "San Francisco, CA", "Seattle, WA", "Austin, TX", "New York, NY", "Chicago, IL",
    "Boston, MA", "Denver, CO", "Atlanta, GA", "Los Angeles, CA", "Dallas, TX"
]
skills_pool = [
    "Python", "Java", "C++", "Go", "Scala", "JavaScript", "React", "Node.js", "Docker", "Kubernetes",
    "AWS", "Azure", "GCP", "SQL", "NoSQL", "Spark", "TensorFlow", "PyTorch", "Pandas", "Tableau"
]
industries = [
    "Software", "FinTech", "Healthcare", "Retail", "Media", "Automotive", "Energy", "Education", "Logistics", "Security"
]
employment_types = ["Full-time", "Part-time", "Contract", "Internship"]
experience_levels = ["Entry", "Mid", "Senior", "Lead"]
certs = [
    "AWS Solutions Architect", "Azure Fundamentals", "Scrum Master (CSM)", "CCNA", "Tableau Desktop", "GCP Professional"
]

def pick_some(pool, lo, hi): return random.sample(pool, random.randint(lo, hi))

def compute_listing_competitiveness(salary, num_skills, num_certs, exp_level, industry_demand):
    base = (salary / 20000) + (num_skills / 2) + num_certs + (2 if exp_level in ["Senior", "Lead"] else 0) + industry_demand
    noise = random.uniform(-1, 1)
    return int(max(0, min(10, base / 3 + noise)))

# --- Schema for job listings (with competitiveness_score) ---
schema = T.StructType([
    T.StructField("job_id", T.StringType(), False),
    T.StructField("job_title", T.StringType(), False),
    T.StructField("company", T.StringType(), False),
    T.StructField("location", T.StringType(), True),
    T.StructField("industry", T.StringType(), True),
    T.StructField("employment_type", T.StringType(), True),
    T.StructField("experience_level", T.StringType(), True),
    T.StructField("required_skills", T.ArrayType(T.StringType()), True),
    T.StructField("required_certifications", T.ArrayType(T.StringType()), True),
    T.StructField("salary_usd", T.IntegerType(), True),
    T.StructField("posting_date", T.DateType(), True),
    T.StructField("competitiveness_score", T.IntegerType(), True)
])

# Firebase
from google.oauth2 import service_account
from google.cloud import firestore

# Pull from Databricks Secret
service_account_json = dbutils.secrets.get(scope="firebase", key="service_account_json")
service_account_info = json.loads(service_account_json)
credentials = service_account.Credentials.from_service_account_info(service_account_info)
project_id = service_account_info["project_id"]
fs_client = firestore.Client(project=project_id, credentials=credentials)

rows = []
today = datetime.date.today()

# Pull listings
COLLECTION_NAME = "job_listings"
docs = fs_client.collection(COLLECTION_NAME).stream()

# Pull job listings from database

for doc in docs:
    d = doc.to_dict() or {}

    job_id        = d.get("job_id", doc.id)
    title         = d.get("title")
    company       = d.get("company")
    location      = d.get("location")
    description   = d.get("description")
    requirements  = d.get("requirements") or []
    skills_needed = d.get("skills_needed") or d.get("skills") or []
    industry      = d.get("industry")

    # 🔥 If you want Firestore job listings to ALSO have a competitiveness score:
    competitiveness = d.get("competitiveness_score")

    posted_raw = d.get("posted_date")
    if hasattr(posted_raw, "date"):
        posted_date = posted_raw.date()
    else:
        posted_date = today

    rows.append(Row(
        job_id=job_id,
        title=title,
        company=company,
        location=location,
        description=description,
        requirements=requirements,
        skills_needed=skills_needed,
        industry=industry,
        posted_date=posted_date,
        competitiveness_score=competitiveness
    ))

# Create Spark dataframe to split data and process in parallel
df = spark.createDataFrame(rows)

from pyspark.sql import functions as F
# Sort by competitiveness_score and pull top 1000 to be filtered by user
df_top1000 = (
    df
    .orderBy(F.desc("competitiveness_score"))
    .limit(1000)
)

print("Top 1000 selected:")
df_top1000.show(20, truncate=False)

spark.sql("CREATE DATABASE IF NOT EXISTS db")
spark.sql("USE db")
spark.sql("DROP TABLE IF EXISTS db.job_listings_ranked")

df_top1000.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("job_listings_ranked")

print("Saved top 1000 job listings ranked by competitiveness_score.")



# Randomized job listings for training
for i in range(25):
    title = random.choice(job_titles)
    company = random.choice(companies)
    loc = random.choice(locations)
    industry = random.choice(industries)
    emp_type = random.choice(employment_types)
    exp_level = random.choice(experience_levels)
    skills = pick_some(skills_pool, 5, 10)
    cert_list = pick_some(certs, 0, 2)
    salary = random.randint(70000, 220000)
    posting_date = today - datetime.timedelta(days=random.randint(0, 30))
    industry_demand = random.randint(0, 3)  # Simulate demand for the industry

    competitiveness = compute_listing_competitiveness(
        salary, len(skills), len(cert_list), exp_level, industry_demand
    )

    rows.append(Row(
        job_id=f"JOB-{1000+i}",
        job_title=title,
        company=company,
        location=loc,
        industry=industry,
        employment_type=emp_type,
        experience_level=exp_level,
        required_skills=skills,
        required_certifications=cert_list,
        salary_usd=salary,
        posting_date=posting_date,
        competitiveness_score=competitiveness
    ))

df = spark.createDataFrame(rows, schema=schema)

display(df)

# COMMAND ----------

# (Optional) Pick a catalog/schema (Unity Catalog) or just a database (Hive metastore)
spark.sql("DROP TABLE IF EXISTS ml_data.job_listings")
spark.sql("CREATE DATABASE IF NOT EXISTS ml_data")
spark.sql("USE ml_data")

# Overwrite (or create) the table as Delta
df.write \
  .format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("job_listings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull in generated training data
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ml_data.job_listings

# COMMAND ----------

from pyspark.sql import functions as F

df_mapped = spark.table("ml_data.job_listings").withColumn(
    "competitiveness_score",
    (F.col("competitiveness_score") * 0.1).cast("double")
)

df_mapped.select("job_id", "company", "competitiveness_score").display()

# 1) See raw schema
df_mapped.printSchema()

# 2) Peek at distinct raw values as strings
df_mapped.select("competitiveness_score").distinct().orderBy("competitiveness_score").show(20, truncate=False)

# 3) Check min/max after a clean cast (no labeling yet)
from pyspark.sql import functions as F

df_debug = (
    df_mapped
    .withColumn("score_str", F.col("competitiveness_score").cast("string"))
    .withColumn("score_num", F.regexp_replace(F.col("score_str"), r"[,%\s]", "").cast("double"))
)

df_debug.agg(
    F.count("*").alias("n"),
    F.count("score_num").alias("n_castable"),
    F.min("score_num").alias("min_score"),
    F.max("score_num").alias("max_score")
).show()


# COMMAND ----------

# pyspark 3.x
from pyspark.sql import functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, CountVectorizer, VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# -----------------------------------------------------------------------------
# Assumes you already have df with these columns (from the previous step):
# ['job_id','job_title','company','location','industry','employment_type',
#  'experience_level','required_skills','required_certifications','salary_usd',
#  'posting_date','competitiveness_score']
# -----------------------------------------------------------------------------

# 1) Define the binary label from competitiveness_score (threshold can be tuned)
df_labeled = df_mapped.withColumn(
    "label",
    F.when(F.col("competitiveness_score") >= F.lit(0.5), F.lit(1.0)).otherwise(F.lit(0.0))
)

# 2) Split train/test
fractions = [0.8, 0.2]
for attempt in range(20):
    train_df, test_df = df_labeled.randomSplit(fractions, seed=42 + attempt)
    t_counts = dict(train_df.groupBy("label").count().collect())
    s_counts = dict(test_df.groupBy("label").count().collect())
    if 0.0 in t_counts and 1.0 in t_counts and 0.0 in s_counts and 1.0 in s_counts:
        print(f"Stratification success on attempt {attempt}")
        break
else:
    raise RuntimeError("Could not obtain both classes in both splits after 20 attempts.")

# 3) Feature transformers
# Array/bag features
# Increase minDF to reduce vocabulary size for CountVectorizer
cv_skills = CountVectorizer(
    inputCol="required_skills",
    outputCol="vec_skills",
    minDF=50.0,  # Increased from 20.0 to 50.0
    minTF=1.0
)
cv_certs = CountVectorizer(
    inputCol="required_certifications",
    outputCol="vec_certs",
    minDF=50.0,
    minTF=1.0)

# Categorical features
idx_industry= StringIndexer(inputCol="industry",         outputCol="idx_industry",        handleInvalid="keep")
idx_title   = StringIndexer(inputCol="job_title",        outputCol="idx_job_title",        handleInvalid="keep")
idx_company = StringIndexer(inputCol="company",          outputCol="idx_company",          handleInvalid="keep")
idx_loc     = StringIndexer(inputCol="location",         outputCol="idx_location",         handleInvalid="keep")
idx_emp     = StringIndexer(inputCol="employment_type",  outputCol="idx_employment_type",  handleInvalid="keep")
idx_exp     = StringIndexer(inputCol="experience_level", outputCol="idx_experience_level", handleInvalid="keep")

ohe = OneHotEncoder(
    inputCols=["idx_job_title", "idx_experience_level"],
    outputCols=["ohe_job_title", "ohe_experience_level"],
    handleInvalid="keep"
)

# Numeric feature(s)
assembler = VectorAssembler(
    inputCols=[
        "vec_skills", "vec_certs",
        "ohe_job_title", "ohe_experience_level",
        "salary_usd"
    ],
    outputCol="features"
)

# 4) Classifier
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=10,
    regParam=0.05,
    elasticNetParam=0.5
)

# 5) Pipeline
pipe = Pipeline(stages=[
    cv_skills, cv_certs, idx_industry,
    idx_title, idx_exp, ohe,
    assembler, lr
])

# 6) Train
model = pipe.fit(train_df)
spark_model = model 
# 7) Evaluate (AUC)
pred_test = model.transform(test_df)

evaluator_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
evaluator_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

auc_roc = evaluator_roc.evaluate(pred_test)
auc_pr  = evaluator_pr.evaluate(pred_test)
print(f"AUC-ROC: {auc_roc:.3f} | AUC-PR: {auc_pr:.3f}")

# 8) Extract clean probability column in [0,1] for "competitive"
get_prob = F.udf(lambda v: float(v[1]), T.DoubleType())
pred_test = pred_test.withColumn("competitive_prob", get_prob(F.col("probability")))

# Preview predictions
pred_test.select(
    "job_id", "job_title", "company", "location", "salary_usd",
    "competitive_prob", "prediction", "label"
).orderBy(F.asc("competitive_prob")).show(20, truncate=False)

# 9) (Optional) Save the model for reuse
# model.write().overwrite().save("dbfs:/models/job_listings_classifier_lr")

# 10) (Optional) Write predictions to a Delta table
# spark.sql("CREATE DATABASE IF NOT EXISTS ml_scoring")
# pred_test.select("job_id","competitive_prob","prediction","label").write \
#   .format("delta").mode("overwrite").saveAsTable("ml_scoring.job_listings_competitiveness_predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Model to model registry

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ml;
# MAGIC CREATE SCHEMA IF NOT EXISTS ml.job_artifacts;
# MAGIC CREATE VOLUME IF NOT EXISTS ml.job_artifacts.models;
# MAGIC CREATE VOLUME IF NOT EXISTS ml.job_artifacts.mlflow_runs;
# MAGIC CREATE VOLUME IF NOT EXISTS ml.job_artifacts.code;

# COMMAND ----------

# =========================
# OPTION 1: Use schema metadata to get exact expanded feature names
# =========================
import os, textwrap
import pandas as pd
import mlflow, mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException
from pyspark.sql import Row
from pyspark.ml import PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexerModel, CountVectorizerModel, VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
# ---- CONFIG: edit these 3 names to match your UC objects ----
CATALOG = "ml"
SCHEMA  = "job_artifacts"
VOLUME_MODELS_DIR = f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/models"
VOLUME_ARTIFACT_ROOT = f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/models/mlflow_runs"

WORKSPACE_USER_DIR = "/Workspace/Users/loganramos00@gmail.com"
CODE_DIR = f"{WORKSPACE_USER_DIR}"

fitted_pipeline = spark_model  # <-- this is your trained PipelineModel
# Get the fitted LR stage:
lrm = [s for s in fitted_pipeline.stages if isinstance(s, LogisticRegressionModel)][0]

print("numFeatures:", lrm.numFeatures)
print("intercept:", float(lrm.intercept))
print("nnz coeffs:", sum(1 for x in lrm.coefficients.toArray() if x != 0.0))
print("first 10 coeffs:", lrm.coefficients.toArray()[:10])


# ---- Discover key stages from the fitted pipeline
ohe_models, indexer_labels, cv_models = {}, {}, {}
assembler, lr_model = None, None

for st in fitted_pipeline.stages:
    if isinstance(st, StringIndexerModel):
        indexer_labels[st.getOutputCol()] = st.labels
    elif isinstance(st, OneHotEncoder):
        for in_col, out_col in zip(st.getInputCols(), st.getOutputCols()):
            ohe_models[out_col] = {"input_col": in_col, "drop_last": st.getDropLast()}
    elif isinstance(st, CountVectorizerModel):
        cv_models[st.getOutputCol()] = {"input_col": st.getInputCol(), "vocab": st.vocabulary}
    elif isinstance(st, VectorAssembler):
        assembler = st
    elif isinstance(st, LogisticRegressionModel):
        lr_model = st

assert assembler is not None, "VectorAssembler not found in pipeline."
assert lr_model is not None, "LogisticRegressionModel not found in pipeline."

print(lr_model)
# -------- Build a realistic RAW input example (these are the raw fields your endpoint will accept)
example_pd = pd.DataFrame({
    "job_id": ["JOB-2025"],
    "job_title": ["ML Engineer"],
    "company": ["Techify"],
    "location": ["Seattle, WA"],
    "industry": ["FinTech"],
    "employment_type": ["Full-time"],
    "experience_level": ["Senior"],
    "required_skills": [["Python", "Spark", "TensorFlow", "AWS", "Docker"]],
    "required_certifications": [["AWS Solutions Architect"]],
    "salary_usd": [180000],
    "posting_date": [pd.to_datetime("2025-10-29")],
})

# -------- Use SCHEMA METADATA on the assembler output to get exact per-dimension names
probe_row = Row(
    job_id="JOB-2025",
    job_title="ML Engineer",
    company="Techify",
    location="Seattle, WA",
    industry="FinTech",
    employment_type="Full-time",
    experience_level="Senior",
    required_skills=["Python", "Spark", "TensorFlow", "AWS", "Docker"],
    required_certifications=["AWS Solutions Architect"],
    salary_usd=180000,
    posting_date=datetime.date(2025, 10, 29)
)

probe_df = spark.createDataFrame([probe_row])

vec_col = assembler.getOutputCol()  # usually "features"
fe_df = fitted_pipeline.transform(probe_df).select(vec_col)

attrs_meta = fe_df.schema[vec_col].metadata.get("ml_attr", {}).get("attrs", {})

if not attrs_meta:
    raise RuntimeError(
        "No attribute metadata found on the assembler output column. "
        "Ensure your pipeline preserved ML attrs, or fall back to Option 3 (index-only names)."
    )

pairs = []
for _group_name, group in attrs_meta.items():
    for a in group:
        idx = a["idx"]
        nm = a.get("name", f"f_{idx}")
        pairs.append((idx, nm))

expanded_feature_names = [nm for idx, nm in sorted(pairs, key=lambda x: x[0])]

num_features = lr_model.numFeatures
if len(expanded_feature_names) != num_features:
    raise AssertionError(f"Expanded names {len(expanded_feature_names)} != lr_model.numFeatures {num_features}")

weights = lr_model.coefficients.toArray().tolist()
print(weights)
if len(weights) != len(expanded_feature_names):
    raise AssertionError(f"Weights {len(weights)} != {len(expanded_feature_names)} feature count")

encoder_spec = {
    "ohe": {
        k: {
            "input_col": v["input_col"],
            "drop_last": v["drop_last"],
            "categories": indexer_labels.get(v["input_col"] + "_idx", [])
        } for k, v in ohe_models.items()
    },
    "cv": cv_models,
    "assembler_inputs": assembler.getInputCols(),
    "expanded_feature_names": expanded_feature_names
}

feature_weights = dict(zip(expanded_feature_names, weights))
print(weights)
print(f"✓ Expanded names extracted from schema: {len(expanded_feature_names)} features")

os.makedirs(CODE_DIR, exist_ok=True)
MODULE_FILE = f"{CODE_DIR}/jobs_pyfunc.py"

with open(MODULE_FILE, "w") as f:
    f.write(textwrap.dedent("""
        import pandas as pd
        import numpy as np
        import mlflow.pyfunc

        def _ohe_row(value, categories, drop_last):
            k = len(categories) - (1 if drop_last else 0)
            vec = np.zeros(k, dtype=float)
            if value is None:
                return vec
            try:
                idx = categories.index(value)
                if drop_last and idx == len(categories) - 1:
                    return vec
                adj = idx if (not drop_last or idx < k) else None
                if adj is not None:
                    vec[adj] = 1.0
            except ValueError:
                pass
            return vec

        def _multi_hot_row(values, vocab):
            vec = np.zeros(len(vocab), dtype=float)
            if isinstance(values, list):
                present = set(values)
                for i, term in enumerate(vocab):
                    if term in present:
                        vec[i] = 1.0
            return vec

        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))

        class CompetitivePyFunc(mlflow.pyfunc.PythonModel):
            \"\"\"Featurizes raw JSON at inference using encoder_spec, then scores with feature_weights.\"\"\"

            def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
                mc = context.model_config or {}
                self.feature_weights = mc.get("feature_weights")
                self.encoder_spec = mc.get("encoder_spec")
                if not self.feature_weights or not self.encoder_spec:
                    raise ValueError("model_config must include 'feature_weights' and 'encoder_spec'")

                self.ohe = self.encoder_spec["ohe"]
                self.cv = self.encoder_spec["cv"]
                self.assembler_inputs = self.encoder_spec["assembler_inputs"]
                self.expanded_feature_names = self.encoder_spec["expanded_feature_names"]

                w = pd.Series(self.feature_weights, index=self.expanded_feature_names)
                self.w_vec = w.astype(float).values

            def _featurize(self, df: pd.DataFrame) -> pd.DataFrame:
                rows = []
                for _, r in df.iterrows():
                    feats = []
                    names = []
                    for col in self.assembler_inputs:
                        if col in self.ohe:
                            spec = self.ohe[col]
                            cats = spec["categories"]
                            drop = spec["drop_last"]
                            raw_val = r.get(spec["input_col"], None)
                            vec = _ohe_row(raw_val, cats, drop)
                            feats.append(vec)
                            use_cats = cats[:-1] if drop else cats
                            names.extend([f"{spec['input_col']}__{c}" for c in use_cats])
                        elif col in self.cv:
                            spec = self.cv[col]
                            raw_vals = r.get(spec["input_col"], None)
                            vec = _multi_hot_row(raw_vals, spec["vocab"])
                            feats.append(vec)
                            names.extend([f"{spec['input_col']}__{t}" for t in spec["vocab"]])
                        else:
                            val = r.get(col, 0.0)
                            try:
                                fval = float(val) if pd.notna(val) else 0.0
                            except Exception:
                                fval = 0.0
                            feats.append(np.array([fval], dtype=float))
                            names.append(col)

                    feat_vec = np.concatenate(feats).astype(float)
                    row_series = pd.Series(feat_vec, index=names)
                    rows.append(row_series)

                X = pd.DataFrame(rows)

                for name in self.expanded_feature_names:
                    if name not in X.columns:
                        X[name] = 0.0
                X = X[self.expanded_feature_names]

                return X

            def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
                X = self._featurize(model_input)
                score = X.values.dot(self.w_vec)
                prob = _sigmoid(score)
                return pd.DataFrame({
                    "candidate_id": model_input.get("candidate_id", pd.Series([None]*len(X))),
                    "competitive_score": prob
                })

        mlflow.models.set_model(CompetitivePyFunc())
    """))

print(f"✓ Wrote module: {MODULE_FILE}")

mlflow.set_tracking_uri("databricks")
EXP_NAME = "/Users/loganramos00@gmail.com/job_pyfunc_uc"

try:
    exp_id = mlflow.create_experiment(EXP_NAME, artifact_location=VOLUME_ARTIFACT_ROOT)
    print(f"✓ Created experiment '{EXP_NAME}' with artifact root: {VOLUME_ARTIFACT_ROOT}")
except MlflowException:
    exp = mlflow.get_experiment_by_name(EXP_NAME)
    if exp:
        print(f"ℹ︎ Experiment exists. artifact_location: {exp.artifact_location}")
    else:
        raise

mlflow.set_experiment(EXP_NAME)

conda_env = {
    'name': 'competitive-env',
    'channels': ['defaults'],
    'dependencies': [
        'python=3.12',
        'pip',
        {'pip': [
            'mlflow>=2.9.0',
            'pandas==1.5.3',
            'numpy==1.26.4',
            'psutil==5.9.0'
        ]}
    ]
}

from pyspark.dbutils import DBUtils
dbutils = DBUtils(spark)
dbutils.fs.mkdirs(VOLUME_ARTIFACT_ROOT)
dbutils.fs.mkdirs(VOLUME_MODELS_DIR)

mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(example_pd, pd.DataFrame({"job_id": [], "competitive_score": []}))

with mlflow.start_run() as run:
    run_id = run.info.run_id
    pyfunc_info = mlflow.pyfunc.log_model(
        artifact_path="jobs_pyfunc",
        python_model="jobs_pyfunc.py",
        model_config={
            "feature_weights": feature_weights,
            "encoder_spec": encoder_spec
        },
        conda_env=conda_env,
        input_example=example_pd,
        signature=signature
    )

    pyfunc_uri = f"runs:/{run_id}/jobs_pyfunc"

print("======================================")
print(" Pyfunc logged at:", pyfunc_uri)
print(" Feature weights + encoder_spec saved in model_config.")
print(" Experiment name:", EXP_NAME)
print(" Artifact root:", VOLUME_ARTIFACT_ROOT)
print("======================================")

# COMMAND ----------

print(feature_weights)

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import time

uc_model_name = "main.ml_models.jobs_pyfunc"  # <catalog>.<schema>.<model>

# Point MLflow to the Unity Catalog registry
mlflow.set_registry_uri("databricks-uc")

# Register the just-logged pyfunc
result = mlflow.register_model(model_uri=pyfunc_uri, name=uc_model_name)
model_version = result.version

# (Optional) Wait until the model version is READY
client = MlflowClient()
while True:
    mv = client.get_model_version(name=uc_model_name, version=model_version)
    if mv.status == "READY":
        break
    time.sleep(2)

# Prefer UC aliases instead of stages:
client.set_registered_model_alias(
    name=uc_model_name, alias="prod", version=model_version
)

print(f"Registered {uc_model_name} v{model_version} and set alias 'prod'.")

# COMMAND ----------

import mlflow
import pandas as pd

# Get model_id from previous registration result
from mlflow import MlflowClient

model_id = "m-6a96ecb8043d4454b585e036d0e33e63"

pyfunc_uri = f"models:/{model_id}"
model = mlflow.pyfunc.load_model(pyfunc_uri)

# Example input (same structure as training)
example_input = pd.DataFrame({
    "candidate_id": ["CAND-2025"],
    "full_name": ["Avery Patel"],
    "location": ["Seattle, WA"],
    "education_level": ["Master"],
    "years_experience": [7.0],
    "skills": [["Python","Spark","TensorFlow","AWS","Docker"]],
    "certifications": [["AWS Solutions Architect"]],
    "current_title": ["ML Engineer"],
    "industries": [["Software","FinTech"]],
    "achievements": [["Open source contributor"]]
})

# Run prediction
result = model.predict(example_input)
display(result)