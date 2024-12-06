import PyPDF2
import os
import nltk
from nltk.corpus import stopwords
import re

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)

def extract_resume_text(pdf_path, remove_stopwords=True):
    """
    Extract text from a PDF resume and optionally remove stopwords.
    
    Args:
        pdf_path (str): Full path to the PDF resume file
        remove_stopwords (bool): Whether to remove stopwords from the text
    
    Returns:
        str: Extracted text from the PDF (with or without stopwords)
    """
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    # Check if file is a PDF
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("The file must be a PDF.")
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Initialize an empty string to store text
            resume_text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                resume_text += page.extract_text() + "\n"
            
            # Clean the text (remove extra whitespaces, convert to lowercase)
            resume_text = re.sub(r'\s+', ' ', resume_text).lower().strip()
            
            # Remove stopwords if requested
            if remove_stopwords:
                # Get English stopwords
                stop_words = set(stopwords.words('english'))
                
                # Split the text into words
                words = resume_text.split()
                
                # Remove stopwords
                filtered_words = [word for word in words if word not in stop_words]
                
                # Rejoin the words
                resume_text = ' '.join(filtered_words)
            
            return resume_text
    
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

# Example usage
def main():
    # Specify the path to your PDF resume
    resume_path = 'resume.pdf'
    
    # Extract resume text with stopwords removed
    resume_text_without_stopwords = extract_resume_text(resume_path, remove_stopwords=True)
    
    # Extract resume text with stopwords (for comparison)
    resume_text_with_stopwords = extract_resume_text(resume_path, remove_stopwords=False)
    
    if resume_text_without_stopwords:
        print("Resume Text Without Stopwords (Preview):")
        print(resume_text_without_stopwords[:500] + "...")
        
        print("\n--- Comparison ---")
        
        print("Original Text Word Count:", 
              len(resume_text_with_stopwords.split()))
        print("Filtered Text Word Count:", 
              len(resume_text_without_stopwords.split()))
        
        # Additional processing can be done here
        # For example, you might want to:
        # - Extract most frequent words
        # - Perform named entity recognition
        
if __name__ == "__main__":
    main()