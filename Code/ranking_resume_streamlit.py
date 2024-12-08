import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import PyPDF2
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# Helper functions
def embed_texts(texts, model, tokenizer):
    """Generate embeddings for a list of texts using BERT."""
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
        embeddings.append(output.last_hidden_state[:, 0, :].mean(0).numpy())
    return np.array(embeddings)

def extract_text_from_pdf(file):
    """Extract text content from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def clean_text(text):
    """Clean text by lowercasing and removing non-alphabetical characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def normalize_vectors(vectors):
    """Normalize vectors to unit length."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def rank_resumes(job_embedding, resume_embeddings, filenames, resumes):
    """Rank resumes based on cosine similarity."""
    similarity_scores = cosine_similarity([job_embedding], resume_embeddings)[0]
    ranked_indices = np.argsort(similarity_scores)[::-1]

    ranked_results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        ranked_results.append({
            "rank": rank,
            "file_name": filenames[idx],
            "similarity_score": similarity_scores[idx] * 100,
            "content_preview": resumes[idx][:500]  # Provide a snippet of the content
        })
    return ranked_results

