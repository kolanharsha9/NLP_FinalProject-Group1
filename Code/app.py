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

#%%
import re
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from PyPDF2 import PdfReader
import streamlit as st
from tqdm import tqdm

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
st.header('Checking Resume score')
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


#%%


# Helper functions
def embed_texts(texts, model, tokenizer):
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        chunks = []
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = tokens['input_ids']
        for i in range(0, input_ids.size(1), 512):
            chunk = {key: val[:, i:i+512] for key, val in tokens.items()}
            with torch.no_grad():
                output = model(**chunk)
            chunks.append(output.last_hidden_state[:, 0, :].mean(0).numpy())
        embeddings.append(np.mean(chunks, axis=0))
    return np.array(embeddings)



def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Streamlit App
st.header("Resume Ranking System")
st.subheader("Upload Job Description and Resumes")

# Upload Job Description
uploaded_pdf = st.file_uploader("Upload Job Description (PDF)", type="pdf")

# Upload Resumes
uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF or Text Files, Max 10)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Parameters
if st.button("Rank Resumes"):
    if uploaded_pdf and uploaded_resumes:
        if len(uploaded_resumes) > 10:
            st.error("Please upload a maximum of 10 resumes.")
        else:
            # Extract Job Description
            job_description = extract_text_from_pdf(uploaded_pdf)

            # Load Resumes
            st.write("Processing resumes...")
            resumes = []
            filenames = []
            for resume_file in uploaded_resumes:
                try:
                    if resume_file.name.endswith(".pdf"):
                        content = extract_text_from_pdf(resume_file)
                    elif resume_file.name.endswith(".txt"):
                        content = resume_file.read().decode("utf-8")
                    else:
                        st.error(f"Unsupported file type: {resume_file.name}")
                        continue

                    cleaned_content = clean_text(content)
                    resumes.append(cleaned_content)
                    filenames.append(resume_file.name)
                except Exception as e:
                    st.error(f"Error processing {resume_file.name}: {str(e)}")

            # Load Pretrained BERT
            st.write("Loading BERT model...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")

            # Embed Resumes and Job Description
            st.write("Generating embeddings for resumes...")
            resume_embeddings = embed_texts(resumes, model, tokenizer)
            st.write("Generating embedding for job description...")
            job_desc_embedding = embed_texts([job_description], model, tokenizer)[0]

            # Normalize Embeddings
            resume_embeddings = normalize_vectors(resume_embeddings)
            job_desc_embedding = job_desc_embedding / np.linalg.norm(job_desc_embedding)

            # Calculate Similarity Scores
            similarity_scores = np.dot(resume_embeddings, job_desc_embedding)

            # Rank Resumes by Similarity Scores
            ranked_indices = np.argsort(similarity_scores)[::-1]

            # Display Ranked Resumes
            st.write("Ranked Resumes Based on Similarity Scores:")
            for rank, idx in enumerate(ranked_indices, start=1):
                st.write(f"### Rank {rank}")
                st.write(f"**File Name:** {filenames[idx]}")
                st.write(f"**Similarity Score:** {similarity_scores[idx] * 100:.2f}%")
                st.write(f"**Resume Content:** {resumes[idx][:500]}...")
                st.write("---")
    else:
        st.error("Please upload a job description PDF and at least one resume.")









#gives you json for resume and jobdes

import streamlit as st
from app4 import process_documents, save_to_json

def main():
    # Streamlit UI components
    st.title("Resume and Job Description Parser")
    
    # API Key input
    api_key = st.text_input("Enter Gemini API Key")
    
    # File uploaders
    resume_file = st.file_uploader("Upload Resume PDF", type=['pdf'])
    job_desc_file = st.file_uploader("Upload Job Description", type=['txt'])
    
    if st.button("Parse Documents"):
        # Save uploaded files temporarily
        with open("temp_resume.pdf", "wb") as f:
            f.write(resume_file.getbuffer())
        
        with open("temp_job_desc.txt", "wb") as f:
            f.write(job_desc_file.getbuffer())
        
        # Process documents
        parsed_resume, parsed_job_description = process_documents(
            api_key, 
            "temp_resume.pdf", 
            "temp_job_desc.txt"
        )
        
        # Display or save results
        if parsed_resume:
            st.json(parsed_resume)
            save_to_json(parsed_resume, 'parsed_resume.json')
        
        if parsed_job_description:
            st.json(parsed_job_description)
            save_to_json(parsed_job_description, 'parsed_job_description.json')

if __name__ == "__main__":
    main()


