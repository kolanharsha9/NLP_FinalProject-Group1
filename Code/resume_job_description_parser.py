import os
import json
import PyPDF2
import google.generativeai as genai
import chardet

def configure_gemini_api(api_key):
    """
    Configure and return the Gemini API model
    
    :param api_key: Google Gemini API key
    :return: Configured Gemini model
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF file
    
    :param pdf_path: Path to the PDF file
    :return: Extracted text from the PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()
        return full_text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def detect_and_read_file(file_path):
    """
    Detect file encoding and read text file
    
    :param file_path: Path to the text file
    :return: File contents as string
    """
    try:
        # First, detect the file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Read the file with detected encoding
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        
        # Fallback methods
        try:
            # Try UTF-8 with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as fallback_error:
            print(f"Fallback reading failed: {fallback_error}")
            return ""


def clean_text(text):
    """
    Clean and normalize text by removing unnecessary whitespace and symbols.

    :param text: Raw text to clean
    :return: Cleaned text
    """
    try:
        # Normalize line breaks and whitespace
        text = text.replace('\r', '').replace('\n', '\n').strip()

        # Remove non-ASCII characters (optional)
        text = ''.join(char for char in text if ord(char) < 128)

        # Remove excessive spaces
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text

def parse_document(model, document_text, document_type):
    """
    Parse document text into structured JSON
    
    :param model: Gemini API model
    :param document_text: Text to be parsed
    :param document_type: Type of document ('resume' or 'job_description')
    :return: Structured information as a dictionary
    """
    # Prompts for resume and job description parsing
    prompts = {
        'resume': f"""
        Parse the following resume text and extract information in a strict JSON format:
        
        Provide these keys:
        - personal_info: Name, contact details, location
        - skills: List of professional and technical skills
        - work_experience: List of work experiences with these sub-keys:
          * company: Company name
          * role: Job title
          * dates: Employment period
          * responsibilities: Key job responsibilities
        - projects: List of significant projects with these sub-keys:
          * name: Project name
          * description: Project description
          * technologies: Technologies used
        - education: List of educational qualifications with these sub-keys:
          * institution: School or university name
          * degree: Degree obtained
          * graduation_year: Year of graduation
        
        If any section is not present in the resume, return an empty list or null.
        
        Strictly return the result as a valid JSON object.

        Resume Text:
        {document_text}
        """,
        'job_description': f"""
        Parse the following job description and extract information in a strict JSON format:
        
        Provide these keys:
        - job_title: Exact job title
        - company: Company name
        - location: Job location
        - employment_type: Full-time, Part-time, Contract, etc.
        - salary_range: Estimated salary range if mentioned
        - required_skills: List of technical and soft skills required
        - responsibilities: Detailed list of job responsibilities
        - qualifications: Educational and experience requirements
        - preferred_qualifications: Additional nice-to-have skills or experiences
        
        If any section is not present in the job description, return an empty list or null.
        
        Strictly return the result as a valid JSON object.

        Job Description Text:
        {document_text}
        """
    }

    # Validate document type
    if document_type not in prompts:
        raise ValueError("Invalid document type. Must be 'resume' or 'job_description'.")

    try:
        # Generate the response
        response = model.generate_content(prompts[document_type])
        
        # Extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            parsed_data = json.loads(json_match.group(0))
        else:
            # If no JSON found, try to parse the entire response
            parsed_data = json.loads(response.text)
        
        return parsed_data

    except Exception as e:
        print(f"Error parsing {document_type}: {e}")
        print(f"Full response text: {response.text}")
        return None

def save_to_json(data, output_path):
    """
    Save parsed data to a JSON file
    
    :param data: Parsed document data
    :param output_path: Path to save the JSON file
    :return: Path to the saved JSON file
    """
    if data:
        try:
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
            
            print(f"Document parsed and saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving JSON: {e}")
    
    return None


def process_documents(api_key, resume_pdf_path, job_description_pdf_path):
    """
    Main function to process resume and job description

    :param api_key: Google Gemini API key
    :param resume_pdf_path: Path to resume PDF
    :param job_description_pdf_path: Path to job description PDF
    :return: Tuple of parsed resume and job description
    """
    try:
        # Configure Gemini API
        model = configure_gemini_api(api_key)

        # Extract text from resume PDF
        resume_text = extract_pdf_text(resume_pdf_path)

        # Extract text from job description PDF
        job_description_text = extract_pdf_text(job_description_pdf_path)

        # Parse resume
        parsed_resume = parse_document(model, resume_text, 'resume')

        # Parse job description
        parsed_job_description = parse_document(model, job_description_text, 'job_description')

        return parsed_resume, parsed_job_description

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


# Example usage (can be removed or commented out when imported)
def main():
    # Replace with your actual Gemini API key
    api_key = 'AIzaSyDw1PTBcbK09IYvQkUI7Fp39A8M1NMm-Pg'

    # Paths for resume PDF and job description PDF file
    resume_pdf_path = 'resume.pdf'
    job_description_pdf_path = 'Job_description.pdf'

    # Process documents
    parsed_resume, parsed_job_description = process_documents(api_key, resume_pdf_path, job_description_pdf_path)

    # Save parsed documents to JSON
    if parsed_resume:
        save_to_json(parsed_resume, 'parsed_resume.json')
    if parsed_job_description:
        save_to_json(parsed_job_description, 'parsed_job_description.json')


if __name__ == "__main__":
    main()
