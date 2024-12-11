import os
import kagglehub
from dataextractor.utils.Utils import find_path, read_json
from dataextractor.ResumeProcessor import ResumeProcessor
from dataextractor.JobDescriptionProcessor import JobDescriptionProcessor


cwd = find_path("JsonConverter")
PROCESSED_RESUMES_PATH = os.path.join(cwd, "Data", "Processed", "Resumes/")
PROCESSED_JOB_DESC_PATH = os.path.join(cwd, "Data", "Processed", "JobDescription/")


def get_filenames_from_dir(directory):
    """
    Retrieve all file names in the specified directory.
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def display_processed_resumes():
    """
    Display extracted details from all processed resumes.
    """
    filenames = get_filenames_from_dir(PROCESSED_RESUMES_PATH)

    for resume in filenames:
        try:
            # Read the JSON file for each resume
            resume_dict = read_json(os.path.join(PROCESSED_RESUMES_PATH, resume))

            # Extract keywords or any other relevant details
            resume_keywords = resume_dict.get("extracted_keywords", [])
            print(f"Processing resume: {resume}")
            print(f"Extracted Keywords: {resume_keywords}")
        except Exception as e:
            print(f"Error processing resume {resume}: {str(e)}")


def display_processed_job_descriptions():
    """
    Display extracted details from all processed job descriptions.
    """
    filenames = get_filenames_from_dir(PROCESSED_JOB_DESC_PATH)

    for job_desc in filenames:
        try:
            # Read the JSON file for each job description
            job_desc_dict = read_json(os.path.join(PROCESSED_JOB_DESC_PATH, job_desc))

            # Extract relevant details or metadata
            job_title = job_desc_dict.get("job_title", "Unknown Title")
            skills_required = job_desc_dict.get("required_skills", [])
            print(f"Processing job description: {job_desc}")
            print(f"Job Title: {job_title}")
            print(f"Required Skills: {skills_required}")
        except Exception as e:
            print(f"Error processing job description {job_desc}: {str(e)}")

def download_job_description_dataset():
    """
    Download the job description dataset from Kaggle.
    """
    print("Downloading job description dataset from Kaggle...")
    path = kagglehub.dataset_download("ravindrasinghrana/job-description-dataset")
    print(f"Dataset downloaded to: {path}")
    return path


if __name__ == "__main__":
    # Process resumes
    print("Processing resumes...")
    resume_processor = ResumeProcessor("InferencePrince555/Resume-Dataset")
    resume_success = resume_processor.process()

    if resume_success:
        print("Resumes processed successfully.")
        # display_processed_resumes()
    else:
        print("Resume processing failed.")

    # Process job descriptions
    print("\nProcessing job descriptions...")
    dataset_path = download_job_description_dataset()

    # Process all files in the downloaded dataset directory
    for file in os.listdir(dataset_path):
        if file.endswith((".pdf", ".json", ".csv", ".txt")):  # Supported formats
            print(f"Processing job description: {file}")
            job_desc_processor = JobDescriptionProcessor(file, dataset_path)
            job_desc_success = job_desc_processor.process()

            if job_desc_success:
                print(f"Job description processed successfully: {file}")
                # display_processed_job_descriptions()
            else:
                print(f"Failed to process job description: {file}")
