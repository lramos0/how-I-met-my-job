# Databricks notebook source
# MAGIC %md
# MAGIC ## Generate Mock Data

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC %pip install --upgrade "supabase"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# pyspark==3.x
from pyspark.sql import SparkSession, Row, types as T
import random, datetime

spark = SparkSession.builder.appName("synthetic_job_matching_profiles_labeled").getOrCreate()
random.seed(42)

# --- (same vocabularies as before, shortened here for brevity) ---
first_names = ["Avery","Jordan","Taylor","Riley","Casey","Skyler","Alex","Sam","Morgan","Quinn"]
last_names  = ["Nguyen","Patel","Garcia","Smith","Kim","Chen","Brown","Johnson","Williams","Jones"]
skills_pool = ["Python","Java","C++","Go","Scala","JavaScript","React","Node.js","Docker","Kubernetes",
               "AWS","SQL","NoSQL","Spark","TensorFlow","PyTorch","Pandas","Tableau","Statistics","Linux"]
roles = ["Software Engineer","Data Scientist","ML Engineer","Product Manager","Designer","DevOps Engineer"]
education_levels = ["High School","Associate","Bachelor","Master","PhD"]
industries = ["Software","FinTech","Healthcare","Retail","Media","Automotive"]
certs = ["AWS Solutions Architect","Azure Fundamentals","Scrum Master (CSM)","CCNA","Tableau Desktop"]
cities = ["San Francisco, CA","Seattle, WA","Austin, TX","New York, NY","Chicago, IL"]

def pick_some(pool, lo, hi): return random.sample(pool, random.randint(lo, hi))
def years_exp_by_education(edu): return {"High School":(0,8),"Associate":(1,10),"Bachelor":(1,12),"Master":(3,15),"PhD":(4,18)}[edu]
def current_title_from_skills(skills):
    if {"TensorFlow","PyTorch"} & set(skills): return "ML Engineer"
    if {"React","Node.js"} & set(skills): return "Software Engineer"
    if {"Docker","Kubernetes"} & set(skills): return "DevOps Engineer"
    if {"Tableau","Statistics"} & set(skills): return "Data Scientist"
    return random.choice(roles)

def compute_competitiveness(years, edu, num_skills, num_certs, num_achievements):
    base = (years / 2) + (num_skills / 2) + num_certs + (2 if edu in ["Master","PhD"] else 0) + num_achievements
    noise = random.uniform(-1, 1)
    return int(max(0, min(10, base / 3 + noise)))

# --- Schema (with competitiveness_score) ---
schema = T.StructType([
    T.StructField("candidate_id", T.StringType(), False),
    T.StructField("full_name", T.StringType(), False),
    T.StructField("location", T.StringType(), True),
    T.StructField("education_level", T.StringType(), True),
    T.StructField("years_experience", T.IntegerType(), True),
    T.StructField("skills", T.ArrayType(T.StringType()), True),
    T.StructField("certifications", T.ArrayType(T.StringType()), True),
    T.StructField("current_title", T.StringType(), True),
    T.StructField("industries", T.ArrayType(T.StringType()), True),
    T.StructField("achievements", T.ArrayType(T.StringType()), True),
    T.StructField("competitiveness_score", T.IntegerType(), True),
    T.StructField("last_updated", T.DateType(), True)
])

rows = []
today = datetime.date.today()

for i in range(25):
    fn, ln = random.choice(first_names), random.choice(last_names)
    edu = random.choice(education_levels)
    yrs = random.randint(*years_exp_by_education(edu))
    skills = pick_some(skills_pool, 6, 12)
    cert_list = pick_some(certs, 0, 2)
    achievements = pick_some(["Mentored juniors","Open source contributions","Conference speaker",
                              "Patent filed","Process automation initiative","Top performer award"], 0, 3)
    title = current_title_from_skills(skills)
    industries_sample = pick_some(industries, 1, 2)

    competitiveness = compute_competitiveness(yrs, edu, len(skills), len(cert_list), len(achievements))

    rows.append(Row(
        candidate_id=f"CAND-{1000+i}",
        full_name=f"{fn} {ln}",
        location=random.choice(cities),
        education_level=edu,
        years_experience=yrs,
        skills=skills,
        certifications=cert_list,
        current_title=title,
        industries=industries_sample,
        achievements=achievements,
        competitiveness_score=competitiveness,
        last_updated=today
    ))

df = spark.createDataFrame(rows, schema=schema)

# Preview
df.show(10, truncate=False)


# COMMAND ----------

import supabase
SUPABASE_URL = "https://xaooqthrquigpwpsbmss.supabase.co"
SUPABASE_KEY = 'yomasaho'
supabase_client: supabase.Client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

import requests
import json

SUPABASE_URL = "https://xaooqthrquigpwpsbmss.supabase.co"
API_KEY = SUPABASE_KEY

headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
}

resp = requests.get(
    f"{SUPABASE_URL}/rest/v1/user_resumes",
    headers=headers,
    params={"select": "*"},
)
resp.raise_for_status()

data = resp.json()   # list[dict] or []

# If Supabase returns a single object instead of list, normalize:
if isinstance(data, dict):
    data = [data]

# Use Pandas as a bridge (works great on serverless; no sparkContext)
pdf = pd.DataFrame(data)
supabase_spark_df = spark.createDataFrame(pdf)

supabase_spark_df.printSchema()
supabase_spark_df.show(10, truncate=False)

joined = df.join(
    supabase_spark_df,
    df.candidate_id == supabase_spark_df.candidate_id,
    "outer"
)

display(joined)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull in generated training data
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

df_mapped = spark.table("ml_data.candidate_profiles").withColumn(
    "competitiveness_score",
    (F.col("competitiveness_score") * 0.1).cast("double")
)

df_mapped.select("full_name", "competitiveness_score").display()

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

from pyspark.sql import functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml.feature import HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 1) Define the binary label from competitiveness_score (threshold can be tuned)
df_labeled = df_mapped.withColumn(
    "label",
    F.when(F.col("competitiveness_score") >= F.lit(0.5), F.lit(1.0)).otherwise(F.lit(0.0))
)

# 2) Split train/test (same as before)
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
# --- Hash-based text features (fixed size) ---
hash_skills = HashingTF(
    inputCol="skills",
    outputCol="skills_vec",
    numFeatures=1024
)

hash_certs = HashingTF(
    inputCol="certifications",
    outputCol="certs_vec",
    numFeatures=512
)

hash_industries = HashingTF(
    inputCol="industries",
    outputCol="industries_vec",
    numFeatures=512
)

hash_achievements = HashingTF(
    inputCol="achievements",
    outputCol="achievements_vec",
    numFeatures=1024
)

# --- Categorical features ---
idx_edu   = StringIndexer(inputCol="education_level", outputCol="idx_education_level", handleInvalid="keep")
idx_title = StringIndexer(inputCol="current_title",   outputCol="idx_current_title",   handleInvalid="keep")
idx_loc   = StringIndexer(inputCol="location",        outputCol="idx_location",        handleInvalid="keep")

ohe = OneHotEncoder(
    inputCols=["idx_education_level", "idx_current_title", "idx_location"],
    outputCols=["ohe_education_level", "ohe_current_title", "ohe_location"],
    handleInvalid="keep"
)

# --- Numeric + assembled features ---
assembler = VectorAssembler(
    inputCols=[
        "skills_vec",
        "certs_vec",
        "industries_vec",
        "achievements_vec",
        "ohe_education_level",
        "ohe_current_title",
        "ohe_location",
        "years_experience"
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

# 5) Pipeline: use hash_* transformers here
pipe = Pipeline(stages=[
    hash_skills, hash_certs, hash_industries, hash_achievements,
    idx_edu, idx_title, idx_loc, ohe,
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
    "candidate_id", "full_name", "current_title", "years_experience",
    "education_level", "competitive_prob", "prediction", "label"
).orderBy(F.asc("competitive_prob")).show(20, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Model to model registry

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ml;
# MAGIC CREATE SCHEMA IF NOT EXISTS ml.talent_artifacts;
# MAGIC CREATE VOLUME IF NOT EXISTS ml.talent_artifacts.mlflow_runs;
# MAGIC CREATE VOLUME IF NOT EXISTS ml.talent_artifacts.code;

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
SCHEMA  = "talent_artifacts"
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
    "candidate_id": "CAND-2025",
    "full_name": "Avery Patel",
    "location": "Seattle, WA",
    "education_level": "Master",
    "years_experience":7.0,
    "skills": [["Python","Spark","TensorFlow","AWS","Docker"]],
    "certifications": [["AWS Solutions Architect"]],
    "current_title": ["ML Engineer"],
    "industries": [["Software","FinTech"]],
    "achievements": [["Open source contributor"]]
})

# -------- Use SCHEMA METADATA on the assembler output to get exact per-dimension names
probe_row = Row(
    candidate_id="CAND-2025",
    full_name="Avery Patel",
    location="Seattle, WA",
    education_level="Master",
    years_experience=7.0,
    skills=["Python","Spark","TensorFlow","AWS","Docker"],
    certifications=["AWS Solutions Architect"],
    current_title="ML Engineer",
    industries=["Software","FinTech"],
    achievements=["Open source contributor"]
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
MODULE_FILE = f"{CODE_DIR}/competitive_pyfunc.py"

with open(MODULE_FILE, "w") as f:
    f.write(textwrap.dedent("""
        import pandas as pd
        import numpy as np
        import mlflow.pyfunc
        from supabase import create_client, Client

        SUPABASE_URL = 'https://xaooqthrquigpwpsbmss.supabase.co'
        SUPABASE_KEY = 'yomasaho'
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

                # 🔧 One-shot reindex: add missing columns + correct order, fill with 0.0
                X = X.reindex(columns=self.expanded_feature_names, fill_value=0.0)

                return X


            def _json_safe(self, obj):
                \"\"\"Recursively convert numpy types to JSON-serializable Python types.\"\"\"
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, list):
                    return [self._json_safe(v) for v in obj]
                if isinstance(obj, dict):
                    return {k: self._json_safe(v) for k, v in obj.items()}
                return obj

            def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
                # 1) Featurize and compute scores
                X = self._featurize(model_input)
                score = X.values.dot(self.w_vec)
                prob = _sigmoid(score)

                # 2) Build JSON-safe records for Supabase
                # Replace NaNs with None (JSON-friendly)
                clean_df = model_input.where(pd.notnull(model_input), None)
                records = clean_df.to_dict(orient="records")
                records = [self._json_safe(r) for r in records]

                # 3) Insert into Supabase, but don't let failures break scoring
                try:
                    if records:
                        supabase_client.table("user_resumes").insert(records).execute()
                except Exception as e:
                    # In serving, this will show up in logs but won't break the endpoint
                    print(f"Supabase insert failed: {e}")

                # 4) Return predictions as a DataFrame
                n = len(X)
                candidate_ids = (
                    model_input["candidate_id"]
                    if "candidate_id" in model_input.columns
                    else pd.Series([None] * n)
                )

                return pd.DataFrame({
                    "candidate_id": candidate_ids,
                    "competitive_score": prob
                })

        mlflow.models.set_model(CompetitivePyFunc())
    """))


print(f"✓ Wrote module: {MODULE_FILE}")

mlflow.set_tracking_uri("databricks")
EXP_NAME = "/Users/loganramos00@gmail.com/competitive_pyfunc_only_uc"

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
            'pandas',
            'numpy',
            'psutil==5.9.0',
            'supabase'
        ]}
    ]
}

from pyspark.dbutils import DBUtils
dbutils = DBUtils(spark)
dbutils.fs.mkdirs(VOLUME_ARTIFACT_ROOT)
dbutils.fs.mkdirs(VOLUME_MODELS_DIR)

mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(example_pd, pd.DataFrame({"candidate_id": [], "competitive_score": []}))

with mlflow.start_run() as run:
    run_id = run.info.run_id
    pyfunc_info = mlflow.pyfunc.log_model(
        artifact_path="competitive_pyfunc",
        python_model="competitive_pyfunc.py",
        model_config={
            "feature_weights": feature_weights,
            "encoder_spec": encoder_spec
        },
        conda_env=conda_env,
        input_example=example_pd,
        signature=signature
    )

    pyfunc_uri = f"runs:/{run_id}/competitive_pyfunc"

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

uc_model_name = "main.ml_models.competitive_pyfunc"  # <catalog>.<schema>.<model>

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

pyfunc_uri = f"models:/m-c22e7c0031054a1b8641a7bfb29f6050"
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



# COMMAND ----------

# Run prediction
result = model.predict(example_input)
print(result)