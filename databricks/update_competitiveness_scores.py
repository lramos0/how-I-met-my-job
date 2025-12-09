# Databricks notebook source
# MAGIC %md
# MAGIC # Update Competitiveness Scores (Batch)
# MAGIC 
# MAGIC This script runs periodically to:
# MAGIC 1. Fetch `job_listings` and `user_resumes` from Supabase where `competitiveness_score` is NULL.
# MAGIC 2. Load the trained MLflow models (`job-difficulty-evaluator` and `applicant-evaluator`).
# MAGIC 3. Infer scores for the new data.
# MAGIC 4. Update Supabase with the calculated scores.

# COMMAND ----------

# MAGIC %pip install supabase mlflow pandas

# COMMAND ----------

import os
import json
import mlflow
import pandas as pd
from supabase import create_client, Client
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# --- Configuration ---
# BEST PRACTICE: Use Databricks Secrets for these values in production
# SUPABASE_URL = dbutils.secrets.get(scope="my-scope", key="supabase-url")
# SUPABASE_KEY = dbutils.secrets.get(scope="my-scope", key="supabase-key")

# Placeholder for development (User to configure)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

# Model URI placeholders - Replace with your actual Model Registry URIs
JOB_MODEL_URI = "models:/job-difficulty-evaluator/Production"
APPLICANT_MODEL_URI = "models:/applicant-evaluator/Production"

# COMMAND ----------

def get_supabase_client() -> Client:
    if "YOUR_SUPABASE" in SUPABASE_URL:
        print("⚠ Warning: Supabase credentials not configured.")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_unscored_jobs(supabase):
    """Fetches job listings that possess a NULL competitiveness_score."""
    response = supabase.table("job_listings") \
        .select("*") \
        .is_("competitiveness_score", "null") \
        .execute()
    return response.data

def fetch_unscored_resumes(supabase):
    """Fetches resumes that possess a NULL competitiveness_score."""
    response = supabase.table("user_resumes") \
        .select("*") \
        .is_("competitiveness_score", "null") \
        .execute()
    return response.data

def update_scores_in_supabase(supabase, table_name, id_col, updates):
    """
    Batch updates rows in Supabase.
    updates: list of dicts, e.g. [{'id': 123, 'competitiveness_score': 7}]
    """
    if not updates:
        return
    
    # Supabase bulk upsert is efficient
    data, count = supabase.table(table_name).upsert(updates).execute()
    print(f"Updated {len(updates)} rows in {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Job Scoring Logic

# COMMAND ----------

supabase = get_supabase_client()

if supabase:
    # 1. Fetch Data
    jobs_data = fetch_unscored_jobs(supabase)
    print(f"Found {len(jobs_data)} unscored jobs.")
    
    if jobs_data:
        # 2. Prepare Data for Model
        # Map Supabase JSON fields to expected Model Schema (pandas DataFrame)
        # Note: Adjust column names to match EXACTLY what your model expects
        df_jobs = pd.DataFrame(jobs_data)
        
        # Example transformations if needed (e.g. Ensure salary is int)
        # df_jobs['salary_usd'] = df_jobs['salary_usd'].fillna(0).astype(int)

        # 3. Load Model & Predict
        try:
            print(f"Loading job model from {JOB_MODEL_URI}...")
            job_model = mlflow.pyfunc.load_model(JOB_MODEL_URI)
            
            # Predict returns a result (likely numeric score or probability)
            predictions = job_model.predict(df_jobs)
            
            # If predictions are a separate DataFrame/Series, extract values
            # Assuming model returns a 'competitiveness_score' or single series
            if isinstance(predictions, pd.DataFrame):
                scores = predictions.iloc[:, 0].tolist() # Take first column
            else:
                scores = predictions.tolist()
                
            # 4. Prepare Updates
            job_updates = []
            for job, score in zip(jobs_data, scores):
                # Scale logic: ensure score is 0-10 integer if that's what DB expects
                # If model output is 0.0-1.0 prob, multiply by 10
                final_score = int(float(score) * 10) if score <= 1.0 else int(score) 
                
                job_updates.append({
                    "id": job["id"], # Assuming 'id' is the primary key
                    "competitiveness_score": final_score
                })
            
            # 5. Write Back
            update_scores_in_supabase(supabase, "job_listings", "id", job_updates)
            
        except Exception as e:
            print(f"Error scoring jobs: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Resume Scoring Logic

# COMMAND ----------

if supabase:
    # 1. Fetch Data
    resumes_data = fetch_unscored_resumes(supabase)
    print(f"Found {len(resumes_data)} unscored resumes.")
    
    if resumes_data:
        df_resumes = pd.DataFrame(resumes_data)
        
        try:
            print(f"Loading applicant model from {APPLICANT_MODEL_URI}...")
            # applicant_model = mlflow.pyfunc.load_model(APPLICANT_MODEL_URI) 
            # Note: Uncomment above when model is ready. 
            
            # TODO: Implement Resume Prediction Logic similar to Jobs above
            # For now, we leave as placeholder until Applicant model is fully registered
            
            print("Applicant scoring skipped (Model pipeline pending).")
            
        except Exception as e:
            print(f"Error scoring resumes: {str(e)}")
