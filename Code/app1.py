import streamlit as st
import os
import sys

# Add the directory containing job_recommendation to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from job_recommendation.job_recommendation import SemanticJobRecommender

def main():
    # Set page configuration
    st.set_page_config(page_title="Job Recommendation Assistant", page_icon="🔍", layout="centered")
    
    # Title and description
    st.title("📋 Job Recommendation Assistant")
    st.markdown("Upload or paste your resume to get personalized job recommendations!")
    
    # Initialize the job recommender
    recommender = SemanticJobRecommender()
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                            ["Paste Resume", "Upload Resume"])
    
    # Resume text variable
    resume_text = ""
    
    if input_method == "Paste Resume":
        # Text area for pasting resume with clear placeholder and height
        resume_text = st.text_area(
            "Paste your resume text here:", 
            height=300, 
            placeholder="Copy and paste your entire resume content here..."
        )
    else:
        # File uploader for resume
        uploaded_file = st.file_uploader(
            "Upload Resume", 
            type=['txt', 'pdf', 'docx'], 
            help="Upload your resume in txt, pdf, or docx format"
        )
        
        # Handle file upload
        if uploaded_file is not None:
            # Basic text extraction (you might want to improve this for different file types)
            try:
                resume_text = uploaded_file.getvalue().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Recommendation button
    if st.button("Get Job Recommendations", type="primary"):
        # Validate resume text
        if resume_text.strip():
            try:
                # Get job recommendations
                recommended_jobs = recommender.recommend_jobs(resume_text)
                
                # Display recommendations
                if recommended_jobs:
                    st.subheader("Top Job Recommendations")
                    for i, job in enumerate(recommended_jobs, 1):
                        with st.expander(f"Job {i}: {job['title']}"):
                            st.write(f"**Job ID:** {job['job_id']}")
                            st.write(f"**Similarity Score:** {job['similarity_score']:.2%}")
                            st.write(f"**Skill Match Ratio:** {job['skill_match_ratio']:.2%}")
                            st.write(f"**Combined Score:** {job['combined_score']:.2%}")
                            
                            st.markdown("**Skills Required:**")
                            st.write(", ".join(job['skills_required']))
                            
                            st.write(f"**Experience Required:** {job['experience_required']['min_years']} - {job['experience_required']['max_years']} years")
                            st.write(f"**Qualifications:** {job['qualifications']}")
                else:
                    st.warning("No job recommendations found.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter or upload a resume first.")

if __name__ == "__main__":
    main()