import os
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from torch.nn import DataParallel
from src.utils.file_utils import load_json
from src.preprocess.resume_processor import preprocess_resume
from src.preprocess.job_desc_processor import preprocess_job_desc

def preprocess_pipeline(resume_dir, job_desc_dir, output_dir, batch_size=1000):
    """
    Preprocess all resumes and job descriptions in batches.
    Save tokenized outputs to files for downstream processing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cleanup existing files in the output directory
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Cleaned up existing files in the output directory: {output_dir}")

    # List all files in resume and job description directories
    resume_files = [os.path.join(resume_dir, f) for f in os.listdir(resume_dir) if f.endswith(".json")]
    job_desc_files = [os.path.join(job_desc_dir, f) for f in os.listdir(job_desc_dir) if f.endswith(".json")]

    print(f"Found {len(resume_files)} resume files.")
    print(f"Found {len(job_desc_files)} job description files.")

    if not resume_files:
        print("No resume files found. Exiting.")
        return

    if not job_desc_files:
        print("No job description files found. Exiting.")
        return

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model = DataParallel(model)  # Distribute model across GPUs
    model = model.to("cuda")


    # Process resumes
    print("Processing Resumes...")
    for i in tqdm(range(0, len(resume_files), batch_size), desc="Resumes", unit="batch"):
        batch = resume_files[i:i + batch_size]
        try:
            processed_resumes = [preprocess_resume(load_json(file)) for file in batch]

            # Tokenization on multiple GPUs
            inputs = tokenizer(processed_resumes, padding=True, truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.cpu()  # Move back to CPU

            torch.save(embeddings, os.path.join(output_dir, f"resume_batch_{i // batch_size}.pt"))
        except Exception as e:
            print(f"Error processing resume batch {i // batch_size}: {e}")
            continue  # Proceed to the next batch even if there is an error

    # Process job descriptions
    print("Processing Job Descriptions...")
    for i in tqdm(range(0, len(job_desc_files), batch_size), desc="Job Descriptions", unit="batch"):
        batch = job_desc_files[i:i + batch_size]
        try:
            processed_job_descs = [preprocess_job_desc(load_json(file)) for file in batch]

            # Tokenization on multiple GPUs
            inputs = tokenizer(processed_job_descs, padding=True, truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.cpu()  # Move back to CPU

            torch.save(embeddings, os.path.join(output_dir, f"job_desc_batch_{i // batch_size}.pt"))
        except Exception as e:
            print(f"Error processing job description batch {i // batch_size}: {e}")
            continue  # Proceed to the next batch even if there is an error