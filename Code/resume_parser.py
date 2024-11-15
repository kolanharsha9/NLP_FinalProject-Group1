# resume_parser.py
import os
import re
from typing import Dict, List, Tuple
import spacy
from pathlib import Path

class ResumeParser:
    def __init__(self):
        # Load spaCy model for NER and text processing
        self.nlp = spacy.load("en_core_web_lg")
        
        # Common sections in resumes
        self.sections = [
            "work experience",
            "education",
            "skills",
            "responsibilities",
            "projects"
        ]

    def parse_resume_file(self, file_path: str) -> Dict:
        """Parse a single resume file and its corresponding label file."""
        try:
            # Read resume content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Read corresponding label file
            label_file = file_path.replace('.txt', '.lab')
            with open(label_file, 'r', encoding='utf-8') as file:
                job_title = file.read().strip()
            
            # Clean and parse content
            cleaned_content = self._clean_text(content)
            parsed_data = self._parse_content(cleaned_content)
            
            return {
                'resume_id': os.path.basename(file_path).replace('.txt', ''),
                'job_title': job_title,
                'raw_text': content,
                'cleaned_text': cleaned_content,
                'parsed_sections': parsed_data,
                'extracted_skills': self._extract_skills(cleaned_content),
                'extracted_experience': self._extract_experience(cleaned_content)
            }
        
        except Exception as e:
            print(f"Error parsing resume {file_path}: {str(e)}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize resume text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,()-]', '', text)
        
        return text.strip()

    def _parse_content(self, text: str) -> Dict:
        """Parse resume content into structured sections."""
        sections = {}
        current_section = "summary"
        current_content = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            lower_line = line.lower()
            if any(section in lower_line for section in self.sections):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = lower_line
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text using spaCy NER and pattern matching."""
        doc = self.nlp(text)
        skills = set()
        
        # Extract technical skills (customize patterns based on your needs)
        skill_patterns = [
            r'(?i)(python|java|sql|c\+\+|javascript|html|css|aws|azure)',
            r'(?i)(machine learning|data science|artificial intelligence)',
            r'(?i)(agile|scrum|waterfall)',
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                skills.add(match.group().lower())
        
        return list(skills)

    def _extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience details."""
        experience_list = []
        experience_pattern = r'(\w+ \d{4})\s*to\s*(\w+ \d{4}|Present)'
        
        matches = re.finditer(experience_pattern, text)
        for match in matches:
            experience_list.append({
                'start_date': match.group(1),
                'end_date': match.group(2)
            })
        
        return experience_list