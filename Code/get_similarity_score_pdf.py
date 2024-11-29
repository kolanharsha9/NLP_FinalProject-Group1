import re
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from PyPDF2 import PdfReader

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
def compute_similarity(job_desc_text, resume_text):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Get embeddings for job description and resume
    job_desc_embedding = get_bert_embedding(job_desc_text, tokenizer, model)
    resume_embedding = get_bert_embedding(resume_text, tokenizer, model)

    # Compute similarity using cosine similarity
    similarity = np.dot(job_desc_embedding, resume_embedding) / (
            np.linalg.norm(job_desc_embedding) * np.linalg.norm(resume_embedding)
    )

    return similarity
# Example file paths
job_desc_pdf = "Processed_jd/job_desc_front_end_engineer.pdf"
resume_pdf = "processed_resume/New-York-Resume-Template-Creative.pdf"

# Extract and clean text
job_desc_text = clean_text(extract_text_from_pdf(job_desc_pdf))
resume_text = clean_text(extract_text_from_pdf(resume_pdf))

# Compute similarity score
similarity_score = compute_similarity(job_desc_text, resume_text)

# Print the score
print(f"Similarity score: {similarity_score:.4f}")
