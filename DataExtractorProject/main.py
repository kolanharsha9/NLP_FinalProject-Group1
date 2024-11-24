import os
from dataextractor.utils.Utils import find_path, read_json
from dataextractor.ResumeProcessor import ResumeProcessor  # Ensure this path is correct

# Set the path for processed resumes
cwd = find_path("DataExtractorProject")
PROCESSED_RESUMES_PATH = os.path.join(cwd, "Data", "Processed", "Resumes/")


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


if __name__ == "__main__":
    # Initialize the ResumeProcessor
    processor = ResumeProcessor("InferencePrince555/Resume-Dataset")

    # Process the dataset and generate JSON files
    success = processor.process()

    if success:
        print("Resumes processed successfully.")
        # Display details from the processed resumes
    else:
        print("Resume processing failed.")




