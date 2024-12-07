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
    AWS_ACCESS_KEY_ID = "ASIATP2NJQDCYS4B2AR6"
    AWS_SECRET_ACCESS_KEY = "t+xAk085Nnm5Mb7SteZyZIccqv5wUl+hUyjDsnMH"
    AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEJT//////////wEaCXVzLWVhc3QtMSJHMEUCIQCyhFs2NB8H8vC0n25TjJDRCucyaSrITZK+b6TrkDcSpgIgIyY1O6+4ltQUwMoeUQzbEygDH67ye80F3gXMprQ0StQqmgMITRABGgwyNDAxNDM0MDExNTciDG2pGLzlqEu4cAnjQSr3AuOgmGQ1XgHd5c6MMlptMdV5ggX76b7kTonkNmmOblG8gQaW0KSdNMHqh82FvGLc05L8vRIdOKeN0djeNe/KREkZnxScMT3bg7o/Pzg8HxL/5WkkaU3EjWQKUbtNcOGpfYUqyBwJYnPDDOc2tW56Z2KJZWDdUbh+OVfTz/I7DqfTDJ1CbljkQezhBvHRB6bHmT8VpK/idZXWgii8ksxAWbyNlR7rebds+VEnjcc509J8riTeXQhBj7FQUYFRXlmGmE2XA4E9ORVfY61pD9rKqSBY/VBL/AKForTmpM/my5fwZt9ZWREj7jjJt96Gfakx+lfmqKmQT4oTGM7rsaCvn1AvF3GoBWcuLNQe0lI8vMu460CsDfjdMo1EF1Gj0HHTD88ribWC5PGSoh0iGnAUl1JqkwHa+Q72X7VsNUZQorZI3j2LL+Fl0Auitid8uIV8qfksVopmPf90gPkAV86gDJZhAagArkYRd5TnC7FBYbAyLqk752c6mDCAyNK6BjqmAdLA/HDqkjou0PQg8XAT2sh+y9Oj++j5YVX1BxfuApABsf1tDGDrPyWyPAq5OPM849b8SonsbrkvgEaKGJpPLYGbWlR+PHeNw6sHcE042AJ/7ZImjdQIhNhyH6ztvhjf1XNRa6kT8DO73YFbjW5e1hF3U4Ldmvo+35Z0MCPM1+jji8IN/k/eoCmcCu2TpB1spsOlXNOm1Zmtfgno4oxs0MgCyuq3kIc="

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