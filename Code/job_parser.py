# job_parser.py
import pandas as pd
from typing import Dict, List
import json

import re
import pandas as pd
from typing import Dict, List
import json

class JobParser:
    def __init__(self):
        self.essential_columns = [
            'Job Id', 'Experience', 'Qualifications', 'Job Title',
            'Job Description', 'skills', 'Responsibilities'
        ]

    def parse_job_descriptions(self, csv_path: str) -> List[Dict]:
        """Parse job descriptions from CSV file."""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Validate essential columns
            missing_cols = [col for col in self.essential_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing essential columns: {missing_cols}")
            
            # Process each job posting
            parsed_jobs = []
            for _, row in df.iterrows():
                parsed_job = self._parse_job_posting(row)
                if parsed_job:
                    parsed_jobs.append(parsed_job)
            
            return parsed_jobs
        
        except Exception as e:
            print(f"Error parsing job descriptions: {str(e)}")
            return []

    def _parse_job_posting(self, row: pd.Series) -> Dict:
        """Parse individual job posting."""
        try:
            # Parse skills (assuming it's a string representation of a list)
            if isinstance(row['skills'], str):
                skills = [skill.strip() for skill in row['skills'].split(',')]
            else:
                skills = []

            # Parse experience requirement
            experience_years = self._parse_experience(row['Experience'])
            
            # Construct structured job posting
            return {
                'job_id': str(row['Job Id']),
                'title': row['Job Title'],
                'experience_required': experience_years,
                'qualifications': row['Qualifications'],
                'description': row['Job Description'],
                'skills_required': skills,
                'responsibilities': row['Responsibilities'],
                'parsed_description': self._parse_description(row['Job Description'])
            }
        
        except Exception as e:
            print(f"Error parsing job posting {row.get('Job Id', 'unknown')}: {str(e)}")
            return None

    def _parse_experience(self, experience_text: str) -> Dict:
        """Parse experience requirement text."""
        try:
            # Handle patterns like "5 to 15 Years"
            pattern = r'(\d+)\s*to\s*(\d+)\s*Years?'
            match = re.search(pattern, str(experience_text))
            
            if match:
                return {
                    'min_years': int(match.group(1)),
                    'max_years': int(match.group(2))
                }
            return None
        except:
            return None

    def _parse_description(self, description: str) -> Dict:
        """Parse job description into structured format."""
        return {
            'full_text': description,
            'length': len(description),
            'key_terms': self._extract_key_terms(description)
        }

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Add custom logic to extract important terms
        # This is a simplified version
        key_terms = []
        important_patterns = [
            r'required',
            r'must have',
            r'essential',
            r'preferred'
        ]
        
        for pattern in important_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                # Get the surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                key_terms.append(text[start:end].strip())
        
        return key_terms