# code/main.py
import os
import pandas as pd
import json

from job_parser import JobParser

def test_parsers():
    # Initialize parsers
    
    job_parser = JobParser()
    
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    jobs_file = os.path.join(base_dir, 'data', 'job_descriptions.csv')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(base_dir, 'job_recommendation')
    os.makedirs(output_dir, exist_ok=True)
    
   
    
    # Process job descriptions
    print("\nProcessing job descriptions...")
    full_jobs_data = job_parser.parse_job_descriptions(jobs_file)
    
    # Extract only essential job data
    jobs_data = []
    for job in full_jobs_data:
        essential_job = {
            'job_id': job['job_id'],
            'title': job['title'],
            'experience_required': job['experience_required'],
            'qualifications': job['qualifications'],
            'skills_required': job['skills_required'],
            'responsibilities': job['responsibilities']
        }
        jobs_data.append(essential_job)
    
    # Save parsed data to JSON files
    
    jobs_json_path = os.path.join(output_dir, 'jobs_data.json')
    
    try:
        
        
        # Save jobs data
        with open(jobs_json_path, 'w', encoding='utf-8') as f:
            json.dump(jobs_data, f, indent=2, ensure_ascii=False)
        print(f"Jobs data saved to: {jobs_json_path}")
        
    except Exception as e:
        print(f"Error saving JSON files: {str(e)}")
    
    # Print summary
    print("\nProcessing Summary:")
   
    print(f"Total jobs processed: {len(jobs_data)}")
    
    
    
    if jobs_data:
        print("\nSample Job Data:")
        sample_job = jobs_data[0]
        print(f"Job ID: {sample_job['job_id']}")
        print(f"Title: {sample_job['title']}")
        print(f"Required Skills: {', '.join(sample_job['skills_required'])}")
    
    return jobs_data

if __name__ == "__main__":
    jobs_data = test_parsers()