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


import re
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from PyPDF2 import PdfReader
import streamlit as st

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Function to get BERT embedding for a text
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to compute similarity score
def compute_similarity(job_desc_text, resume_text, tokenizer, model):
    # Get embeddings for job description and resume
    job_desc_embedding = get_bert_embedding(job_desc_text, tokenizer, model)
    resume_embedding = get_bert_embedding(resume_text, tokenizer, model)

    # Compute similarity using cosine similarity
    similarity = np.dot(job_desc_embedding, resume_embedding) / (
        np.linalg.norm(job_desc_embedding) * np.linalg.norm(resume_embedding)
    )

    return similarity

# Integration in the middle of a larger Streamlit script
st.header("Upload Files")
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Compute Similarity"):
    if job_desc_file is not None and resume_file is not None:
        # Load BERT tokenizer and model
        st.write("Loading BERT model...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Extract and preprocess text
        st.write("Processing files...")
        job_desc_text = clean_text(extract_text_from_pdf(job_desc_file))
        resume_text = clean_text(extract_text_from_pdf(resume_file))

        # Compute similarity score
        st.write("Computing similarity...")
        similarity_score = compute_similarity(job_desc_text, resume_text, tokenizer, model)

        # Display result
        st.success(f"Similarity Score: {similarity_score:.4f}")
    else:
        st.error("Please upload both the job description and the resume.")








