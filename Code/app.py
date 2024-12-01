import streamlit as st
import pandas as pd


st.title("Resume and Job Matcher")

st.sidebar.title("Menu")
menu_options = ["Home", "About"]
choice = st.sidebar.selectbox("Select an option", menu_options)

if choice == "Home":
    st.header("Job Description")
    job_description = st.text_area("Paste the job description here")

    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Drag and drop your resume here", type=["pdf", "docx"])

    if uploaded_file is not None:
        st.write("Uploaded Resume:", uploaded_file.name)
        st.header("Output Results")
        st.write("Results will be displayed here after processing.")

elif choice == "About":
    st.header("About")
    st.write("This is a basic Streamlit application for matching resumes with job descriptions.")








