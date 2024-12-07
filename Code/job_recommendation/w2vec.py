import os
import json
import re
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

class SemanticJobRecommender:
    def __init__(self, 
                 jobs_data_path=None, 
                 embeddings_path=None, 
                 vector_size=100, 
                 window=5, 
                 min_count=1):
        """
        Initialize the semantic job recommender with Word2Vec embeddings.
        
        :param jobs_data_path: Path to jobs JSON file
        :param embeddings_path: Path to save/load embeddings
        :param vector_size: Dimension of word vectors
        :param window: Context window size
        :param min_count: Minimum word frequency to be included
        """
        # Set up paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Default paths if not provided
        if jobs_data_path is None:
            jobs_data_path = os.path.join(script_dir, 'jobs_data.json')
        
        if embeddings_path is None:
            embeddings_path = os.path.join(script_dir, 'job_embeddings.model')
        
        # Load job data
        with open(jobs_data_path, 'r', encoding='utf-8') as f:
            self.jobs_data = json.load(f)
        
        # Parameters
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.embeddings_path = embeddings_path
        
        # Load or create embeddings
        self.model = self._load_or_create_embeddings()
    
    def _advanced_text_preprocessing(self, text):
        """
        Advanced text preprocessing technique.
        
        :param text: Input text
        :return: Cleaned and tokenized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common stop words and generic terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = [word for word in text.split() if word not in stop_words]
        
        return tokens
    
    def _prepare_training_data(self):
        """
        Prepare job descriptions for Word2Vec training.
        
        :return: List of tokenized job descriptions
        """
        training_data = []
        for job in self.jobs_data:
            # Combine multiple text sources
            full_text = (
                f"{job['title']} "
                f"Skills: {' '.join(job.get('skills_required', []))} "
                f"Qualifications: {job.get('qualifications', '')} "
                f"Responsibilities: {job.get('responsibilities', '')}"
            )
            
            # Preprocess and tokenize
            job_tokens = self._advanced_text_preprocessing(full_text)
            training_data.append(job_tokens)
        
        return training_data
    
    def _load_or_create_embeddings(self):
        """
        Load existing embeddings or create new ones.
        
        :return: Trained Word2Vec model
        """
        # Try to load existing embeddings
        if os.path.exists(self.embeddings_path):
            try:
                print("Loading existing Word2Vec embeddings...")
                return Word2Vec.load(self.embeddings_path)
            except Exception as e:
                print(f"Error loading embeddings: {e}. Creating new embeddings.")
        
        # Create new embeddings
        print("Creating new Word2Vec embeddings...")
        training_data = self._prepare_training_data()
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=training_data, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=4
        )
        
        # Save the model
        print(f"Saving embeddings to {self.embeddings_path}")
        model.save(self.embeddings_path)
        
        return model
    
    def _document_vector(self, tokens):
        """
        Create document vector by averaging word vectors.
        
        :param tokens: List of tokens
        :return: Document vector
        """
        # Filter tokens present in the model's vocabulary
        vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
        
        # Return zero vector if no words found
        if not vectors:
            return np.zeros(self.vector_size)
        
        # Average word vectors
        return np.mean(vectors, axis=0)
    
    def recommend_jobs(self, resume_text, top_n=5):
        """
        Recommend jobs using Word2Vec semantic matching.
        
        :param resume_text: Raw resume text
        :param top_n: Number of top recommendations to return
        :return: List of recommended jobs
        """
        # Preprocess resume
        resume_tokens = self._advanced_text_preprocessing(resume_text)
        
        # Create resume vector
        resume_vector = self._document_vector(resume_tokens)
        
        # Compute job vectors
        job_vectors = []
        for job in self.jobs_data:
            # Combine job text sources
            full_text = (
                f"{job['title']} "
                f"Skills: {' '.join(job.get('skills_required', []))} "
                f"Qualifications: {job.get('qualifications', '')} "
                f"Responsibilities: {job.get('responsibilities', '')}"
            )
            
            # Tokenize and vectorize
            job_tokens = self._advanced_text_preprocessing(full_text)
            job_vector = self._document_vector(job_tokens)
            job_vectors.append(job_vector)
        
        # Compute cosine similarities
        similarities = [cosine_similarity([resume_vector], [job_vector])[0][0] 
                        for job_vector in job_vectors]
        
        # Create scored recommendations
        job_scores = []
        for idx, similarity in enumerate(similarities):
            job = self.jobs_data[idx]
            
            # Calculate skill match ratio
            job_skills = set(skill.lower() for skill in job.get('skills_required', []))
            resume_skill_set = set(resume_tokens)
            skill_match_ratio = len(job_skills.intersection(resume_skill_set)) / len(job_skills) if job_skills else 0
            
            # Combined score (weighted)
            combined_score = (0.7 * similarity) + (0.3 * skill_match_ratio)
            
            job_scores.append({
                'job': job,
                'similarity_score': similarity,
                'skill_match_ratio': skill_match_ratio,
                'combined_score': combined_score
            })
        
        # Sort and prepare recommendations
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
                    'skills_required': job.get('skills_required', []),
                    'experience_required': job.get('experience_required', {}),
                    'qualifications': job.get('qualifications', '')
                }
                recommendations.append(recommendation)
                seen_titles.add(job['title'])
                
                if len(recommendations) == top_n:
                    break
        
        return recommendations

    @staticmethod
    def read_pdf_resume(pdf_path):
        """
        Read text content from a PDF resume.
        
        :param pdf_path: Path to the PDF resume
        :return: Extracted text from the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                resume_text = []
                for page in pdf_reader.pages:
                    resume_text.append(page.extract_text())
                
                # Join text from all pages
                return ' '.join(resume_text)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

def main():
    recommender = SemanticJobRecommender()
    
    pdf_path = 'resume.pdf'
    
    if not os.path.exists(pdf_path):
        print("resume.pdf not found. Exiting.")
        return
    
    resume_text = recommender.read_pdf_resume(pdf_path)
    
    if not resume_text.strip():
        print("No resume text extracted. Exiting.")
        return
    
    recommended_jobs = recommender.recommend_jobs(resume_text)
    
    # Sort recommendations by Similarity Score in descending order
    recommended_jobs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print("\n--- Advanced Job Recommendations (Ranked by Similarity Score) ---")
    for i, job in enumerate(recommended_jobs, 1):
        print(f"\n{i}. Job Title: {job['title']}")
        print(f"   Job ID: {job['job_id']}")
        print(f"   Similarity Score: {job['similarity_score']:.2%}")
        print(f"   Skill Match Ratio: {job['skill_match_ratio']:.2%}")
        print(f"   Combined Score: {job['combined_score']:.2%}")
        print(f"   Skills Required: {', '.join(job['skills_required'])}")
        print(f"   Experience Required: {job.get('experience_required', {}).get('min_years', 'N/A')} - {job.get('experience_required', {}).get('max_years', 'N/A')} years")
        print(f"   Qualifications: {job['qualifications']}")

if __name__ == "__main__":
    main()