from resume_job_description_parser import process_documents
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_weighted_similarity(resume, job_description, weights):
    """
    Calculates a weighted similarity score between resume and job description fields.

    Args:
        resume (dict): Parsed resume JSON.
        job_description (dict): Parsed job description JSON.
        weights (dict): Weights for each section (skills, responsibilities, qualifications).

    Returns:
        float: Final weighted similarity score.
    """
    # Extract relevant sections
    resume_skills = " ".join(resume.get('skills', []))
    resume_responsibilities = " ".join(
        [" ".join(exp.get('responsibilities', [])) for exp in resume.get('work_experience', [])]
    )
    resume_qualifications = " ".join([edu.get('degree', '') for edu in resume.get('education', [])])

    job_skills = " ".join(job_description.get('required_skills', []))
    job_responsibilities = " ".join(job_description.get('responsibilities', []))
    job_qualifications = " ".join(job_description.get('qualifications', []))

    # Generate embeddings
    resume_embeddings = {
        'skills': model.encode(resume_skills, convert_to_tensor=True),
        'responsibilities': model.encode(resume_responsibilities, convert_to_tensor=True),
        'qualifications': model.encode(resume_qualifications, convert_to_tensor=True)
    }
    job_embeddings = {
        'skills': model.encode(job_skills, convert_to_tensor=True),
        'responsibilities': model.encode(job_responsibilities, convert_to_tensor=True),
        'qualifications': model.encode(job_qualifications, convert_to_tensor=True)
    }

    # Calculate cosine similarities
    similarities = {
        field: util.cos_sim(resume_embeddings[field], job_embeddings[field]).item()
        for field in weights.keys()
    }

    # Compute weighted average similarity score
    weighted_similarity = sum(similarities[field] * weights[field] for field in weights)
    return weighted_similarity


# Main code
if __name__ == "__main__":
    api_key = "AIzaSyDw1PTBcbK09IYvQkUI7Fp39A8M1NMm-Pg"
    resume_file = "resume.pdf"  # Replace with the actual resume file path
    job_description_file = "Job-Description.pdf"  # Replace with the actual job description file path

    # Parse the documents
    parsed_resume, parsed_job_description = process_documents(api_key, resume_file, job_description_file)

    # Define weights for each field
    weights = {
        'skills': 0.5,
        'responsibilities': 0.4,
        'qualifications': 0.1
    }

    # Calculate the final similarity score
    final_similarity_score = calculate_weighted_similarity(parsed_resume, parsed_job_description, weights)
    print(f"Final Similarity Score: {final_similarity_score:.2f}")
