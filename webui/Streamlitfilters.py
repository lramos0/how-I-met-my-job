import streamlit as st

# Load job data
df = pd.read_csv('job_postings.csv')

# Sidebar filters
company = st.sidebar.selectbox("Company", options=["All"] + df['company_name'].unique().tolist())
title = st.sidebar.selectbox("Job Title", options=["All"] + df['job_title'].unique().tolist())

# Apply filters
filtered_df = df.copy()
if company != "All":
    filtered_df = filtered_df[filtered_df['company_name'] == company]
if title != "All":
    filtered_df = filtered_df[filtered_df['job_title'] == title]

# Display results
st.write(filtered_df.sort_values(by='match_score', ascending=False).head(10))