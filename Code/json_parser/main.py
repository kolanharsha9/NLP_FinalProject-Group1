# code/main.py
#%%

import os
import pandas as pd
import json
from resume_parser import ResumeParser
from job_parser import JobParser

#%%

def test_parsers():
    # Initialize parsers
    resume_parser = ResumeParser()
    job_parser = JobParser()
    
    # Set up paths
    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir= os.path.abspath('../..')
    resume_dir = os.path.join(base_dir,'assests','data','resume_txt')
    jobs_file = os.path.join(base_dir,'assests','data','job_descriptions.csv')

    print(resume_dir)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(base_dir,'assests','data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process resumes
    print("Processing resumes...")
    resumes_data = []
    for filename in sorted(os.listdir(resume_dir)):
        if filename.endswith('.txt'):
            file_path = os.path.join(resume_dir, filename)
            print(f"Processing resume: {filename}")
            parsed_resume = resume_parser.parse_resume_file(file_path)
            if parsed_resume:
                # Extract detailed resume data
                detailed_resume = {
                    'resume_id': parsed_resume['resume_id'],
                    'job_title': parsed_resume['job_title'],
                    'skills': {
                        'technical_skills': parsed_resume['extracted_skills'],
                        'all_skills': parsed_resume['parsed_sections'].get('skills', '')
                    },
                    'experience': {
                        'timeline': parsed_resume['extracted_experience'],
                        'details': parsed_resume['parsed_sections'].get('work experience', ''),
                        'projects': parsed_resume['parsed_sections'].get('projects', '')
                    },
                    'education': {
                        'details': parsed_resume['parsed_sections'].get('education', '')
                    },
                    'responsibilities': parsed_resume['parsed_sections'].get('responsibilities', ''),
                    'summary': parsed_resume['parsed_sections'].get('summary', ''),
                    'sections': {
                        section: content
                        for section, content in parsed_resume['parsed_sections'].items()
                        if section not in ['skills', 'work experience', 'education', 'projects', 'responsibilities', 'summary']
                    }
                }
                
                # Process the text in each section to clean up formatting
                for section in detailed_resume:
                    if isinstance(detailed_resume[section], str):
                        detailed_resume[section] = detailed_resume[section].strip().replace('\n\n', '\n')
                    elif isinstance(detailed_resume[section], dict):
                        for key in detailed_resume[section]:
                            if isinstance(detailed_resume[section][key], str):
                                detailed_resume[section][key] = detailed_resume[section][key].strip().replace('\n\n', '\n')
                
                resumes_data.append(detailed_resume)
    
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
    resume_json_path = os.path.join(output_dir, 'resume_data.json')
    jobs_json_path = os.path.join(output_dir, 'jobs_data.json')
    
    try:
        # Save resume data
        with open(resume_json_path, 'w', encoding='utf-8') as f:
            json.dump(resumes_data, f, indent=2, ensure_ascii=False)
        print(f"\nResume data saved to: {resume_json_path}")
        
        # Save jobs data
        with open(jobs_json_path, 'w', encoding='utf-8') as f:
            json.dump(jobs_data, f, indent=2, ensure_ascii=False)
        print(f"Jobs data saved to: {jobs_json_path}")
        
    except Exception as e:
        print(f"Error saving JSON files: {str(e)}")
    
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
        print(f"Technical Skills: {', '.join(sample_resume['skills']['technical_skills'])}")
    
    if jobs_data:
        print("\nSample Job Data:")
        sample_job = jobs_data[0]
        print(f"Job ID: {sample_job['job_id']}")
        print(f"Title: {sample_job['title']}")
        print(f"Required Skills: {', '.join(sample_job['skills_required'])}")
    
    return resumes_data, jobs_data

if __name__ == "__main__":
    resumes_data, jobs_data = test_parsers()