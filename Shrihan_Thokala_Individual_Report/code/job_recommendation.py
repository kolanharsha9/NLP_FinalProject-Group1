import os
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticJobRecommender:
    def __init__(self, jobs_data_path=None):
        """
        Initialize the semantic job recommender with advanced matching techniques.
        """
        # Load job data
        if jobs_data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            jobs_data_path = os.path.join(script_dir, 'jobs_data.json')
        
        with open(jobs_data_path, 'r', encoding='utf-8') as f:
            self.jobs_data = json.load(f)
        
        # Create vectorizer during initialization
        self.vectorizer = TfidfVectorizer()
        
        # Precompute processed job data
        self.processed_jobs, self.job_vectors = self._preprocess_and_vectorize_jobs()
    
    def _advanced_text_preprocessing(self, text):
        """
        Advanced text preprocessing technique.
        
        :param text: Input text
        :return: Cleaned and processed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common stop words and generic terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        return text
    
    def _extract_key_skills(self, text):
        """
        Extract potential key skills from text.
        
        :param text: Input text
        :return: List of potential skills
        """
        # Common skill-related words and patterns
        skill_keywords = [
            'experience', 'proficient', 'expert', 'skilled', 'knowledge', 
            'familiar', 'working', 'understanding', 'competent'
        ]
        
        # Preprocess text
        processed_text = self._advanced_text_preprocessing(text)
        
        # Extract potential skills
        words = processed_text.split()
        skills = []
        
        for i in range(len(words)):
            # Check for skills preceded by skill-related keywords
            if i > 0 and words[i-1] in skill_keywords:
                skills.append(words[i])
            
            # Check for longer skill phrases
            if i < len(words) - 1:
                two_word_skill = f"{words[i]} {words[i+1]}"
                skills.append(two_word_skill)
        
        return list(set(skills))
    
    def _preprocess_and_vectorize_jobs(self):
        """
        Preprocess job descriptions and create TF-IDF vectors.
        
        :return: Tuple of processed job texts and their TF-IDF vectors
        """
        # Prepare job texts with comprehensive information
        processed_job_texts = []
        for job in self.jobs_data:
            job_text = (
                f"{job['title']} "
                f"Skills: {' '.join(job['skills_required'])} "
                f"Qualifications: {job['qualifications']} "
                f"Responsibilities: {job['responsibilities']}"
            )
            processed_job_texts.append(self._advanced_text_preprocessing(job_text))
        
        # Create TF-IDF vectors using the pre-initialized vectorizer
        job_vectors = self.vectorizer.fit_transform(processed_job_texts)
        
        return processed_job_texts, job_vectors
    
    def recommend_jobs(self, resume_text, top_n=5):
        """
        Recommend jobs using advanced semantic matching.
        
        :param resume_text: Raw resume text
        :param top_n: Number of top recommendations to return
        :return: List of recommended jobs
        """
        # Preprocess resume
        processed_resume = self._advanced_text_preprocessing(resume_text)
        resume_skills = self._extract_key_skills(resume_text)
        
        # Vectorize resume using the existing vectorizer
        resume_vector = self.vectorizer.transform([processed_resume])
        
        # Compute cosine similarities
        cosine_similarities = cosine_similarity(resume_vector, self.job_vectors)[0]
        
        # Enhance scoring with skill matching
        job_scores = []
        for idx, similarity_score in enumerate(cosine_similarities):
            job = self.jobs_data[idx]
            
            # Calculate skill match ratio
            job_skills = set(skill.lower() for skill in job['skills_required'])
            resume_skill_set = set(resume_skills)
            skill_match_ratio = len(job_skills.intersection(resume_skill_set)) / len(job_skills) if job_skills else 0
            
            # Combined score (weighted)
            combined_score = (0.7 * similarity_score) + (0.3 * skill_match_ratio)
            
            job_scores.append({
                'job': job,
                'similarity_score': similarity_score,
                'skill_match_ratio': skill_match_ratio,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        job_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Remove duplicates and prepare recommendations
        seen_titles = set()
        recommendations = []
        
        for job_score in job_scores:
            job = job_score['job']
            if job['title'] not in seen_titles:
                recommendation = {
                    'title': job['title'],
                    'job_id': job['job_id'],
                    'similarity_score': job_score['similarity_score'],
                    'skill_match_ratio': job_score['skill_match_ratio'],
                    'combined_score': job_score['combined_score'],
                    'skills_required': job['skills_required'],
                    'experience_required': job['experience_required'],
                    'qualifications': job['qualifications']
                }
                recommendations.append(recommendation)
                seen_titles.add(job['title'])
                
                if len(recommendations) == top_n:
                    break
        
        return recommendations

def main():
    recommender = SemanticJobRecommender()
    
    print("Paste your resume. Press Enter twice when finished:")
    resume_text = []
    while True:
        line = input()
        if line == "":
            break
        resume_text.append(line)
    
    resume_text = '\n'.join(resume_text)
    
    if not resume_text.strip():
        print("No resume text entered. Exiting.")
        return
    
    recommended_jobs = recommender.recommend_jobs(resume_text)
    
    print("\n--- Advanced Job Recommendations ---")
    for i, job in enumerate(recommended_jobs, 1):
        print(f"\n{i}. Job Title: {job['title']}")
        print(f"   Job ID: {job['job_id']}")
        print(f"   Similarity Score: {job['similarity_score']:.2%}")
        print(f"   Skill Match Ratio: {job['skill_match_ratio']:.2%}")
        print(f"   Combined Score: {job['combined_score']:.2%}")
        print(f"   Skills Required: {', '.join(job['skills_required'])}")
        print(f"   Experience Required: {job['experience_required']['min_years']} - {job['experience_required']['max_years']} years")
        print(f"   Qualifications: {job['qualifications']}")

if __name__ == "__main__":
    main()