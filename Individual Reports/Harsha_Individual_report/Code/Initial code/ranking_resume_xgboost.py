import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os
import PyPDF2
from tqdm import tqdm  # For progress bar
from collections import Counter
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score  # Normalized Discounted Cumulative Gain


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


# Generate Features and Labels for Training
def generate_features(resume_embeddings, job_desc_embedding, labels):
    features = []
    relevance_scores = []
    for idx, embedding in enumerate(resume_embeddings):
        similarity = np.dot(embedding, job_desc_embedding)  # Similarity score
        features.append([similarity])  # Add additional features if available
        # Binary relevance: 1 for matching label, 0 otherwise (adjust based on your data)
        relevance_scores.append(1 if labels[idx] == "target_label" else 0)
    return np.array(features), np.array(relevance_scores)


# Train and Evaluate XGBoost Model
def train_xgboost(features, relevance_scores):
    X_train, X_test, y_train, y_test = train_test_split(features, relevance_scores, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg',
        'eta': 0.1,
        'max_depth': 6,
    }
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10)
    return model, X_test, y_test


# Main Pipeline
if __name__ == "__main__":
    # Paths (use your own paths)
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

    # Generate Features and Labels
    features, relevance_scores = generate_features(resume_embeddings, job_desc_embedding, labels)

    # Train XGBoost
    model, X_test, y_test = train_xgboost(features, relevance_scores)

    # Predict Relevance Scores
    dtest = xgb.DMatrix(X_test)
    predicted_scores = model.predict(dtest)

    # Evaluate NDCG
    ndcg = ndcg_score([y_test], [predicted_scores])
    print(f"NDCG Score: {ndcg:.4f}")

    # Rank Resumes
    predicted_scores_full = model.predict(xgb.DMatrix(features))
    ranked_indices = np.argsort(predicted_scores_full)[::-1]
    top_10_indices = ranked_indices[:10]

    # Display Top 10 Resumes
    print("Top 10 Resumes Based on XGBoost Ranking:\n")
    for rank, idx in enumerate(top_10_indices, start=1):
        print(f"Rank {rank}:")
        print(f"File Name: {filenames[idx]}")
        print(f"Predicted Relevance Score: {predicted_scores_full[idx]:.4f}")
        print(f"Label: {labels[idx]}")
        print(f"Resume Content:\n{resumes[idx][:500]}...")
        print("\n" + "-" * 80 + "\n")
