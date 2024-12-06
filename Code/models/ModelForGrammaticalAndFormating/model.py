import boto3
import json
import PyPDF2
import logging
from typing import Dict, List, Any
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BedrockResumeAnalyzer:
    def __init__(
            self,
            aws_access_key_id: str,
            aws_secret_access_key: str,
            aws_session_token: str = None,
            region_name: str = 'us-east-1'
    ):
        """
        Initialize the Bedrock Resume Analyzer with AWS credentials
        """
        try:
            # Create AWS session with provided credentials
            self.session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name
            )

            # Create Bedrock runtime client
            self.bedrock_runtime = self.session.client('bedrock-runtime')
            logger.info("Bedrock Resume Analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Bedrock Resume Analyzer: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file with improved error handling
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = [page.extract_text() for page in pdf_reader.pages]

                # Combine text and remove excessive whitespaces
                cleaned_text = " ".join(full_text).strip()

                logger.info(f"Successfully extracted text from {pdf_path}")
                return cleaned_text

        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise ValueError(f"File not found: {pdf_path}")

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise ValueError(f"PDF extraction error: {e}")

    def analyze_resume_text(self, resume_text: str) -> Dict[str, Any]:
        """
        Analyze the resume text with a more focused prompt
        """
        prompt = f"""Analyze the resume text with precision:

Resume Text:
{resume_text}

Provide a concise analysis focusing on:
1. Major grammatical errors
2. Key formatting inconsistencies
3. Clarity and professional presentation

Detailed Requirements:
- Identify top 3-5 most significant grammatical errors
- List major formatting issues
- Provide constructive recommendations
- Calculate numerical scores for grammar and formatting

Output Format:
- Grammar Score: [0-100]
- Formatting Score: [0-100]
- Top Grammatical Errors:
  - Error Description
- Formatting Issues:
  - Issue Description
- Recommendations:
  - Improvement Suggestion
"""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,  # More focused output
                "top_p": 0.8
            })

            response = self.bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=body
            )

            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']

            logger.info("Resume analysis completed successfully")
            return self._parse_resume_analysis(analysis_text)

        except Exception as e:
            logger.error(f"Resume analysis failed: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "grammar_score": 0,
                "formatting_score": 0,
                "grammatical_errors": [],
                "formatting_issues": [],
                "recommendations": []
            }

    def _parse_resume_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """
        Parse the raw analysis text into a structured format
        """
        try:
            # Enhanced parsing with more robust regex
            grammar_match = re.search(r'Grammar Score: (\d+)', analysis_text)
            formatting_match = re.search(r'Formatting Score: (\d+)', analysis_text)

            return {
                "grammar_score": int(grammar_match.group(1)) if grammar_match else 0,
                "formatting_score": int(formatting_match.group(1)) if formatting_match else 0,
                "grammatical_errors": self._extract_section(analysis_text, 'Top Grammatical Errors'),
                "formatting_issues": self._extract_section(analysis_text, 'Formatting Issues'),
                "recommendations": self._extract_section(analysis_text, 'Recommendations')
            }

        except Exception as e:
            logger.error(f"Parsing analysis failed: {e}")
            return {}

    def _extract_section(self, text: str, section_name: str) -> List[Dict]:
        """
        Extract a specific section from the analysis text
        """
        section_start = text.find(section_name)
        if section_start == -1:
            return []

        section_lines = text[section_start:].split('\n')
        results = []

        for line in section_lines[1:]:
            if not line.startswith('- '):
                break
            results.append({
                "description": line.lstrip('- ').strip(),
                "severity": "medium"
            })

        return results


def analyze_resume(
        aws_access_key_id: str,
        aws_secret_access_key: str,
        resume_path: str,
        aws_session_token: str = None
) -> Dict:
    """
    Main function to analyze a resume file with comprehensive error handling
    """
    try:
        analyzer = BedrockResumeAnalyzer(
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token
        )

        resume_text = analyzer.extract_text_from_pdf(resume_path)
        return analyzer.analyze_resume_text(resume_text)

    except Exception as e:
        logger.error(f"Resume analysis process failed: {e}")
        return {
            "error": str(e),
            "status": "failed"
        }


# Example usage
if __name__ == "__main__":
    # Replace with your actual AWS credentials
    AWS_ACCESS_KEY_ID = "ASIATP2NJQDCZCXPCZQ5"
    AWS_SECRET_ACCESS_KEY = "h4fuP2jTl7InNM0lL9IK0qT9uJj58cOqR3ZpSyxL"
    AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEHwaCXVzLWVhc3QtMSJGMEQCIFWDMG4TXGnMUrjxwK1ygEfQGt20VUnRAsx8anbiKCrTAiBQbAQ37qFtG6HQhEKIjN8oP/7olxvaP7Rrjtn+1dgK+iqaAwg1EAEaDDI0MDE0MzQwMTE1NyIMIJT2wYrwPPsY/q9sKvcCnZRRjcXi3me1TALvOAkz1l5YfU8aCsz0dcdXFaJ0M92nVFK62mwNzkoGofxEDxH6tFQXGS8KIN3aVMuXpurED1y4xITipLkY3h3xfH2zPACrvEhPCw8yFxHTFtk0zN0QQaXBIxx3Y8FQmfqUu7cFJbvzlpwz9NtNm0SRc8YzP5lSQ9AigPZdXG1FzRQu71kXw+rO0g8dciqF+vRASLzZhM8Zf82ndND7DdzqSH90A+mwf9Sf+PIP11m+yPRMA4/E6peqg+pwmaNHLJq4BcwmW3fGPBtlDummxa4fG2xVAnj2uUzNkhDlAAVrQsUJITBP/GczvzmJUiFgeC1YGtYSTcMXdrE3tqaRbIdvnws5zb4CJlvtzJUCQQ7cnLkihr/Yo4/aDiEkedhz3pvc3nqLlD7p/T+2ggnRjAEBvT0lJylef+ZPy40xj/PhzVdH9xKfbMKg+4lX8kvBsa12k14d0PB/oKxHavf9WPQpfA+u0lkAUiM0V4ZzMLahzboGOqcBZ/HWj2QgL+aFrr/CS6VTEbRD+s5190KH3iEbTanSSwvwGkbhAye5jjgCZijqfMBDf/sRg1By9bPN549TaDe/nCLgXfaZmPOYII6yDfDNgFttESVha3rATmPhQQl19fAXx8rF3FNPSvBkgbZIdvjikzMcOGlTyPwxwJ3UtSsjvZeaNE7xCNIQat+MphNKf/igW3HjW8i0RoBGviucwyxCWDrb8iEHY1s="  # Optional, only if using temporary credentials

    # Path to your PDF resume file
    RESUME_PATH = "resumes/ResumeAman (2).pdf"

    # Analyze the resume
    results = analyze_resume(
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        RESUME_PATH,
        AWS_SESSION_TOKEN
    )

    # Print out the analysis results
    print("Resume Analysis Results:")
    print(f"Grammar Score: {results.get('grammar_score', 'N/A')}/100")
    print(f"Formatting Score: {results.get('formatting_score', 'N/A')}/100")

    print("\nGrammatical Errors:")
    for error in results.get('grammatical_errors', []):
        print(f"- {error['description']}")

    print("\nFormatting Issues:")
    for issue in results.get('formatting_issues', []):
        print(f"- {issue['description']}")