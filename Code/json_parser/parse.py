# code/main.py
import os
import pandas as pd
from resume_parser import ResumeParser
from job_parser import JobParser

import re
import pandas as pd
from typing import Dict, List
import json


def detailed_extraction():
    # Initialize parsers
    resume_parser = ResumeParser()
    job_parser = JobParser()

    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resume_dir = os.path.join(base_dir, 'data', 'resume')
    jobs_file = os.path.join(base_dir, 'data', 'job_descriptions.csv')

    # Process resumes one by one
    print("=== RESUME EXTRACTION ===")
    resumes_data = []
    for filename in sorted(os.listdir(resume_dir)):
        if filename.endswith('.txt'):
            file_path = os.path.join(resume_dir, filename)
            parsed_resume = resume_parser.parse_resume_file(file_path)
            
            if parsed_resume:
                print(f"\n--- Resume: {filename} ---")
                print(f"Resume ID: {parsed_resume.get('resume_id', 'N/A')}")
                print(f"Job Title: {parsed_resume.get('job_title', 'N/A')}")
                print(f"Skills: {parsed_resume.get('extracted_skills', [])}")
                print(f"Work Experience: {parsed_resume.get('extracted_experience', [])}")
                print("Parsed Sections:", list(parsed_resume.get('parsed_sections', {}).keys()))
                
                # Optional: Detailed section content
                print("\nSection Details:")
                for section, content in parsed_resume.get('parsed_sections', {}).items():
                    print(f"{section.upper()} (first 200 chars): {content[:200]}...")

    # Process job descriptions
    print("\n=== JOB DESCRIPTION EXTRACTION ===")
    jobs_data = job_parser.parse_job_descriptions(jobs_file)
    for job in jobs_data:
        print(f"\nJob ID: {job.get('job_id', 'N/A')}")
        print(f"Title: {job.get('title', 'N/A')}")
        print(f"Required Skills: {job.get('skills_required', [])}")
        print(f"Experience Required: {job.get('experience_required', 'N/A')}")

if __name__ == "__main__":
    detailed_extraction()