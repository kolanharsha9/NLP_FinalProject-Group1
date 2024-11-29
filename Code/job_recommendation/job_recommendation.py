import os
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class JobRecommender:
    def __init__(self, jobs_data_path=None):
        """
        Initialize the job recommender with job data.
        
        :param jobs_data_path: Path to the JSON file containing job descriptions
        """
        # If no path is provided, try to find jobs_data.json in the current directory
        if jobs_data_path is None:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            jobs_data_path = os.path.join(script_dir, 'jobs_data.json')
        
        # Verify the file exists
        if not os.path.exists(jobs_data_path):
            raise FileNotFoundError(f"Could not find jobs_data.json. Looked in: {jobs_data_path}")
        
        # Load jobs data
        with open(jobs_data_path, 'r', encoding='utf-8') as f:
            self.jobs_data = json.load(f)
        
        # Load spaCy English language model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy English model not found. Please install it using: python -m spacy download en_core_web_sm")
            raise
        
        # Preprocess job data
        self.processed_jobs = self._preprocess_jobs()
    
    def _preprocess_jobs(self):
        """
        Preprocess job descriptions for better matching.
        
        :return: List of processed job descriptions
        """
        processed_jobs = []
        for job in self.jobs_data:
            # Combine relevant job information for comprehensive matching
            job_text = f"{job['title']} Qualifications: {job['qualifications']} " \
                       f"Skills: {' '.join(job['skills_required'])} " \
                       f"Responsibilities: {job['responsibilities']}"
            
            # Remove special characters and convert to lowercase
            job_text = re.sub(r'[^a-zA-Z\s]', '', job_text.lower())
            
            processed_jobs.append(job_text)
        
        return processed_jobs
    
    def _preprocess_resume(self, resume_text):
        """
        Preprocess resume text for matching.
        
        :param resume_text: Raw resume text
        :return: Processed resume text
        """
        # Remove special characters and convert to lowercase
        processed_resume = re.sub(r'[^a-zA-Z\s]', '', resume_text.lower())
        return processed_resume
    
    def recommend_jobs(self, resume_text, top_n=5):
        """
        Recommend top jobs based on resume content.
        
        :param resume_text: Raw resume text
        :param top_n: Number of top recommendations to return
        :return: List of recommended jobs with similarity scores
        """
        # Preprocess resume
        processed_resume = self._preprocess_resume(resume_text)
        
        # Combine processed jobs and resume for vectorization
        all_texts = self.processed_jobs + [processed_resume]
        
        # Use TF-IDF Vectorization for text similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compute cosine similarity
        # The last vector is the resume, so we compare it against all job descriptions
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
        # Get top N job recommendations with their scores
        # Create a list of tuples (index, similarity_score)
        job_indices_scores = list(enumerate(cosine_similarities))
        
        # Sort by similarity score in descending order
        job_indices_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates while maintaining order
        seen_jobs = set()
        recommendations = []
        
        for idx, similarity_score in job_indices_scores:
            job = self.jobs_data[idx]
            
            # Use a combination of job title and skills as a unique identifier
            job_key = (job['title'], tuple(sorted(job['skills_required'])))
            
            if job_key not in seen_jobs:
                recommendation = {
                    'title': job['title'],
                    'job_id': job['job_id'],
                    'similarity_score': similarity_score,
                    'skills_required': job['skills_required'],
                    'experience_required': job['experience_required'],
                    'qualifications': job['qualifications']
                }
                recommendations.append(recommendation)
                seen_jobs.add(job_key)
                
                # Stop when we have top_n unique recommendations
                if len(recommendations) == top_n:
                    break
        
        return recommendations

def main():
    # Initialize the job recommender
    recommender = JobRecommender()
    
    # Get resume text from user input
    print("Please paste your resume text. Press Enter twice when finished:")
    resume_text = []
    while True:
        line = input()
        if line == "":
            # If an empty line is entered, break the input
            break
        resume_text.append(line)
    
    # Join the resume lines
    resume_text = '\n'.join(resume_text)
    
    # Verify resume was entered
    if not resume_text.strip():
        print("No resume text entered. Exiting.")
        return
    
    # Get job recommendations
    recommended_jobs = recommender.recommend_jobs(resume_text)
    
    # Display recommendations
    print("\n--- Top Job Recommendations ---")
    for i, job in enumerate(recommended_jobs, 1):
        print(f"\n{i}. Job Title: {job['title']}")
        print(f"   Job ID: {job['job_id']}")
        print(f"   Similarity Score: {job['similarity_score']:.2%}")
        print(f"   Skills Required: {', '.join(job['skills_required'])}")
        print(f"   Experience Required: {job['experience_required']['min_years']} - {job['experience_required']['max_years']} years")
        print(f"   Qualifications: {job['qualifications']}")

if __name__ == "__main__":
    main()