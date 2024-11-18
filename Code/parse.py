# code/main.py
import os
import pandas as pd
from resume_parser import ResumeParser
from job_parser import JobParser

import re
import pandas as pd
from typing import Dict, List
import json


def test_parsers():
    # Initialize parsers
    resume_parser = ResumeParser()
    job_parser = JobParser()
    
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resume_dir = os.path.join(base_dir, 'data', 'resume')
    jobs_file = os.path.join(base_dir, 'data', 'job_descriptions.csv')
    
    # Process resumes
    print("Processing resumes...")
    resumes_data = []
    for filename in sorted(os.listdir(resume_dir)):
        if filename.endswith('.txt'):
            file_path = os.path.join(resume_dir, filename)
            print(f"Processing resume: {filename}")
            parsed_resume = resume_parser.parse_resume_file(file_path)
            if parsed_resume:
                resumes_data.append(parsed_resume)
    
    # Process job descriptions
    print("\nProcessing job descriptions...")
    jobs_data = job_parser.parse_job_descriptions(jobs_file)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total resumes processed: {len(resumes_data)}")
    print(f"Total jobs processed: {len(jobs_data)}")
    
    # Print sample data
    if resumes_data:
        print("\nSample Resume Data:")
        sample_resume = resumes_data[0]
        print(f"Resume ID: {sample_resume['resume_id']}")
        print(f"Job Title: {sample_resume['job_title']}")
        print(f"Skills: {', '.join(sample_resume['extracted_skills'])}")
    
    if jobs_data:
        print("\nSample Job Data:")
        sample_job = jobs_data[0]
        print(f"Job ID: {sample_job['job_id']}")
        print(f"Title: {sample_job['title']}")
        print(f"Required Skills: {', '.join(sample_job['skills_required'])}")
    
    return resumes_data, jobs_data

if __name__ == "__main__":
    resumes_data, jobs_data = test_parsers()