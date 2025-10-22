import streamlit as st

st.title("Job-Resume Matching Platform")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    # Run matching logic here
    st.success("Resume uploaded and processed.")
    # Show top matches
    st.dataframe(filtepred_df)
