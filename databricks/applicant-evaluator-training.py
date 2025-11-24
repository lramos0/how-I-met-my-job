# Databricks notebook source
# MAGIC %md
# MAGIC ## Generate Mock Data

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]" databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# pyspark==3.x
from pyspark.sql import SparkSession, Row, types as T
import random, datetime

spark = SparkSession.builder.appName("synthetic_job_matching_profiles_labeled").getOrCreate()
random.seed(42)

# // SAME VOCABULARIES AS BEFORE //
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

# // SCHEMA (WITH COMPETITIVENESS_SCORE) //
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

# (Optional) Pick a catalog/schema (Unity Catalog) or just a database (Hive metastore)
spark.sql("DROP TABLE IF EXISTS ml_data.candidate_profiles")
spark.sql("CREATE DATABASE IF NOT EXISTS ml_data")
spark.sql("USE ml_data")

# Overwrite (or create) the table as Delta
df.write \
  .format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("candidate_profiles")

# COMMAND ----------

# // EVALUATE (AUC) //
pred_test = model.transform(test_df)

evaluator_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
evaluator_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

auc_roc = evaluator_roc.evaluate(pred_test)
auc_pr  = evaluator_pr.evaluate(pred_test)
print(f"AUC-ROC: {auc_roc:.3f} | AUC-PR: {auc_pr:.3f}")

# // EXTRACT CLEAN PROBABILITY COLUMN //
get_prob = F.udf(lambda v: float(v[1]), T.DoubleType())
pred_test = pred_test.withColumn("competitive_prob", get_prob(F.col("probability")))

# Preview predictions
pred_test.select(
    "candidate_id", "full_name", "current_title", "years_experience",
    "education_level", "competitive_prob", "prediction", "label"
).orderBy(F.asc("competitive_prob")).show(20, truncate=False)

# // SAVE MODEL (OPTIONAL) //
# model.write().overwrite().save("dbfs:/models/competitive_classifier_lr")

# // WRITE PREDICTIONS (OPTIONAL) //
# spark.sql("CREATE DATABASE IF NOT EXISTS ml_scoring")
# pred_test.select("candidate_id","competitive_prob","prediction","label").write \
#   .format("delta").mode("overwrite").saveAsTable("ml_scoring.candidate_competitiveness_predictions")


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
WORKSPACE_USER_DIR = f"/Workspace/Users/{current_user}"
CODE_DIR = f"{WORKSPACE_USER_DIR}"

fitted_pipeline = spark_model  # <-- this is the trained PipelineModel
# Get the fitted LR stage:
lrm = [s for s in fitted_pipeline.stages if isinstance(s, LogisticRegressionModel)][0]

print("numFeatures:", lrm.numFeatures)
print("intercept:", float(lrm.intercept))
print("nnz coeffs:", sum(1 for x in lrm.coefficients.toArray() if x != 0.0))
print("first 10 coeffs:", lrm.coefficients.toArray()[:10])


# // DISCOVER KEY STAGES FROM THE FITTED PIPELINE //
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
# // BUILD A REALISTIC RAW INPUT EXAMPLE //
example_pd = pd.DataFrame({
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

# // USE SCHEMA METADATA ON THE ASSEMBLER OUTPUT //
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
        "Ensure pipeline preserved ML attrs, or fall back to Option 3 (index-only names)."
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
EXP_NAME = f"/Users/{current_user}/competitive_pyfunc_only_uc"

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy/Update Model Serving Endpoint
# MAGIC Automatically creates or updates a serving endpoint for this model.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

endpoint_name = "applicant-competitiveness-endpoint"
# Use the model name and version from the registration step above
# uc_model_name was defined earlier as "main.ml_models.competitive_pyfunc"
# model_version was defined earlier

print(f"Deploying model {uc_model_name} version {model_version} to endpoint {endpoint_name}...")

try:
    # Check if endpoint exists
    w.serving_endpoints.get(name=endpoint_name)
    print(f"Updating existing endpoint {endpoint_name}...")
    
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                entity_name=uc_model_name,
                entity_version=model_version,
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
except Exception as e:
    # If not found (or other error), try creating
    print(f"Endpoint {endpoint_name} not found (or error: {e}). Creating...")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=uc_model_name,
                    entity_version=model_version,
                    scale_to_zero_enabled=True,
                    workload_size="Small"
                )
            ]
        )
    )

print(f"Endpoint {endpoint_name} deployment initiated.")