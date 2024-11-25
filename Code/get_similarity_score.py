import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to generate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Function to read the content of a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Calculate similarity
def calculate_similarity(embedding1, embedding2):
    embedding1 = embedding1.detach().numpy()
    embedding2 = embedding2.detach().numpy()
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# Main function to process job description and resume
def process_job_and_resume(job_path, resume_path):
    # Read job description and resume
    job_text = read_file(job_path)
    resume_text = read_file(resume_path)

    # Generate embeddings
    job_embedding = get_bert_embedding(job_text)
    resume_embedding = get_bert_embedding(resume_text)

    # Calculate similarity
    similarity_score = calculate_similarity(job_embedding, resume_embedding)

    return similarity_score

# Main script
if __name__ == "__main__":
    job_path = "Processed_jd\JobDescription-job_desc_front_end_engineer.pdf325be79e-5389-4acd-89e1-0f5ab27260fa.json"  # Replace with your job description file path
    resume_path = "processed_resume\Resume-alfred_pennyworth_pm.pdfc6e4644f-cc6c-48e7-9efc-213e97d24865.json"       # Replace with your resume file path

    overall_similarity = process_job_and_resume(job_path, resume_path)

    # Display overall similarity score
    print(f"Overall Similarity Score: {round(overall_similarity * 100, 2)}%")
