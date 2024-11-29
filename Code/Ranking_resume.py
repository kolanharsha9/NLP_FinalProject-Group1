import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os
import PyPDF2
from tqdm import tqdm  # For progress bar
from collections import Counter

# Load Resumes and Labels
def load_data(resume_folder, label_folder, max_resumes=100):
    resumes, labels, filenames = [], [], []
    for filename in os.listdir(resume_folder):
        if filename.endswith(".txt") and len(resumes) < max_resumes:
            resume_id = filename.split('.')[0]
            resume_path = os.path.join(resume_folder, filename)
            try:
                # Try reading with utf-8 encoding
                with open(resume_path, 'r', encoding='utf-8') as f:
                    resumes.append(f.read())
                    filenames.append(filename)
            except UnicodeDecodeError:
                print(f"Warning: Unable to decode {resume_path} with utf-8. Trying latin-1.")
                try:
                    # Fallback to latin-1 encoding
                    with open(resume_path, 'r', encoding='latin-1') as f:
                        resumes.append(f.read())
                        filenames.append(filename)
                except Exception as e:
                    print(f"Error reading {resume_path}: {e}")
                    continue  # Skip this file if it fails again

            # Process label file
            label_file = os.path.join(label_folder, f"{resume_id}.lab")
            try:
                with open(label_file, 'r') as f:
                    labels.append(f.read().strip())
            except FileNotFoundError:
                print(f"Warning: Label file {label_file} not found. Skipping this resume.")
                resumes.pop()  # Remove resume if its label is missing.
                filenames.pop()

    return resumes, labels, filenames

# Embed Texts in Chunks (Handles Long Texts)
def embed_texts(texts, model, tokenizer):
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):  # Progress bar
        chunks = []
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = tokens['input_ids']
        for i in range(0, input_ids.size(1), 512):
            chunk = {key: val[:, i:i+512] for key, val in tokens.items()}
            with torch.no_grad():
                output = model(**chunk)
            chunks.append(output.last_hidden_state[:, 0, :].mean(0).numpy())  # CLS token
        embeddings.append(np.mean(chunks, axis=0))  # Aggregate chunks
    return np.array(embeddings)

# Extract Job Description from PDF
def extract_job_description(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    return " ".join([page.extract_text() for page in reader.pages])

# Normalize Vectors for Cosine Similarity
def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Main Pipeline
if __name__ == "__main__":
    # Paths
    resume_folder = "resumes_corpus"
    label_folder = "resumes_corpus"
    job_desc_pdf = "Processed_jd/sample-resume-for-cyber-security-analyst.pdf"

    # Load Data
    resumes, labels, filenames = load_data(resume_folder, label_folder, max_resumes=100)

    # Print Label Analysis
    label_counts = Counter(labels)
    print("Label Analysis:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    print("\n" + "-" * 80 + "\n")

    # Load Pretrained BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Embed Resumes and Job Description
    resume_embeddings = embed_texts(resumes, model, tokenizer)
    job_description = extract_job_description(job_desc_pdf)
    job_desc_embedding = embed_texts([job_description], model, tokenizer)[0]

    # Normalize Embeddings
    resume_embeddings = normalize_vectors(resume_embeddings)
    job_desc_embedding = job_desc_embedding / np.linalg.norm(job_desc_embedding)

    # Calculate Similarity Scores
    similarity_scores = np.dot(resume_embeddings, job_desc_embedding)

    # Rank Resumes by Similarity Scores
    ranked_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
    top_10_indices = ranked_indices[:10]

    # Display Top 10 Resumes with Similarity Scores and Labels
    print("Top 10 Resumes Based on Similarity Scores:\n")
    for rank, idx in enumerate(top_10_indices, start=1):
        print(f"Rank {rank}:")
        print(f"File Name: {filenames[idx]}")
        print(f"Similarity Score: {similarity_scores[idx] * 100:.2f}%")
        print(f"Label: {labels[idx]}")  # Display the corresponding label
        print(f"Resume Content:\n{resumes[idx][:500]}...")  # Display first 500 characters of the resume
        print("\n" + "-" * 80 + "\n")
