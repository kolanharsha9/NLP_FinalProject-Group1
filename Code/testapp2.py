import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import base64
from models.ModelForGrammaticalAndFormating.model import BedrockResumeAnalyzer
from ranking_resume_streamlit import embed_texts,extract_text_from_pdf,clean_text,normalize_vectors,rank_resumes
from models.resume_train_preprocess_test import gen_resume
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import PyPDF2
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
from io import StringIO
from visualization import generate_word_cloud,generate_sankey,generate_bar_chart,process_documents, generate_heatmap, generate_skill_venn
from resume_job_description_parser import process_documents, extract_pdf_text, clean_text
from sentence_transformers import SentenceTransformer, util
from similarity_score_with_weights import calculate_weighted_similarity
from models.skill_gap import analyze_skill_gap 
from job_recommendation.w2v_fast import SemanticJobRecommender
from dotenv import load_dotenv
import os

@st.cache_data
def parse_documents_once(api_key, resume_file, job_description_text):
    """
    Parses the resume file and job description text only once.

    Args:
        api_key (str): API key for the parser.
        resume_file: Uploaded resume file.
        job_description_text: Job description input as text.

    Returns:
        tuple: Parsed resume and job description as JSON objects.
    """
    # Save resume temporarily
    with open("temp_resume.pdf", "wb") as f:
        f.write(resume_file.getbuffer())

    # Parse documents
    return process_documents(api_key, "temp_resume.pdf", job_description_text)

def main_page():
    st.title("Resume and Job Description Analyzer")

    # Initialize the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    st.subheader("Upload Resume and Paste Job Description")

    # Upload Resume
    resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

    # Paste Job Description
    job_description_text = st.text_area("Paste Job Description Text Here", placeholder="Enter or paste job description...")

    if resume_file and job_description_text.strip():
        st.spinner("Processing documents...")
        api_key = "AIzaSyDw1PTBcbK09IYvQkUI7Fp39A8M1NMm-Pg"  # Replace with your actual API key

        # Clean job description text
        job_description_text = clean_text(job_description_text)

        # Extract and clean resume text
        resume_text = extract_pdf_text(resume_file)
        resume_text = clean_text(resume_text)

        parsed_resume, parsed_job_description = parse_documents_once(api_key, resume_file, job_description_text)

        # Display parsed data or warn if parsing failed
        if parsed_resume and parsed_job_description:
            st.write(parsed_resume)
            st.write(parsed_job_description)
        else:
            st.warning("Failed to parse the resume or job description. Ensure the file format is correct and the text is readable.")

        st.header('Similarity Score for Resume and Job Description')
        st.subheader("Adjust Weights")
        skills_weight = st.slider("Skills Weight", 0.0, 1.0, 0.5, 0.01)
        responsibilities_weight = st.slider("Responsibilities Weight", 0.0, 1.0, 0.4, 0.01)
        qualifications_weight = st.slider("Qualifications Weight", 0.0, 1.0, 0.1, 0.01)

        # Ensure weights sum to 1 with a small tolerance for floating-point errors
        total_weight = skills_weight + responsibilities_weight + qualifications_weight
        tolerance = 1e-6  # Allowable tolerance for floating-point comparison

        if abs(total_weight - 1.0) > tolerance:
            st.warning(f"Total weight is {total_weight:.2f}. Please adjust the sliders to make the total weight 1.0.")
        else:
            # Define weights
            weights = {
                'skills': skills_weight,
                'responsibilities': responsibilities_weight,
                'qualifications': qualifications_weight
            }

            # Calculate similarity score
            final_similarity_score = calculate_weighted_similarity(parsed_resume, parsed_job_description, weights)

            # Display the result
            st.success(f"Final Similarity Score: {final_similarity_score:.2f}")
            st.header('Skill Gap Analysis')
            if st.button("Analyze Skill Gap"):
                skill_gap_analysis = analyze_skill_gap(parsed_resume, parsed_job_description)
                st.write(skill_gap_analysis)
                
            st.header("Select Visualization")
    visualization1 = st.selectbox(
        "Choose a visualization:",
        ["Word Cloud", "Venn Diagram", "Bar Chart", "Heatmap"]
    )

    
    if resume_file and job_description_text.strip():
        resume_text = extract_text_from_pdf(resume_file)
            
            
        documents = [resume_text, job_description_text]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
            
            
        if visualization1 == "Word Cloud":
            st.subheader("Word Clouds")
            generate_word_cloud(tfidf_matrix, feature_names, 0, "Resume Word Cloud")
            generate_word_cloud(tfidf_matrix, feature_names, 1, "Job Description Word Cloud")
        elif visualization1 == "Venn Diagram":
            st.subheader("Venn Diagram")
            generate_skill_venn(parsed_resume, parsed_job_description)
        elif visualization1 == "Bar Chart":
            st.subheader("Bar Chart")
            generate_bar_chart(tfidf_matrix, feature_names)
        elif visualization1 == "Heatmap":
            st.subheader("Heatmap")
            generate_heatmap(parsed_resume, parsed_job_description)
        else:
            st.error("Please upload a resume and enter job description text.")
            
resume_generator = gen_resume()

def resume_generation_page():
    st.title("Resume Generation")
    st.write("Generate a professional resume using our AI-powered tool.")
    user_prompt = st.text_area("Enter your details (e.g., name, experience, skills):", key="resume_gen_prompt")
    model_options = ["models_bart", "models_bart1"]
    selected_model = st.selectbox("Select a Model", model_options)
    generate_button = st.button("Generate Resume")

    if generate_button and user_prompt.strip():
        with st.spinner("Generating your resume..."):
            generated_resume = resume_generator.generate_resume(user_prompt, selected_model)
            st.success("Resume generated successfully!")
            # st.text_area("Generated Resume", generated_resume, height=300)
            st.markdown(f"<pre>{generated_resume}</pre>", unsafe_allow_html=True)
            st.download_button("Download Resume", generated_resume, file_name="generated_resume.txt", mime="text/plain")


def job_recommendation():
    st.header("Job Recommendation System 💼")
    
    # File uploader for resume
    uploaded_resume = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
    
    if uploaded_resume is not None:
        # Save the uploaded PDF
        with open("resume.pdf", "wb") as f:
            f.write(uploaded_resume.getbuffer())
        
        # Create recommender instance
        recommender = SemanticJobRecommender()
        
        # Extract resume text
        resume_text = recommender.read_pdf_resume("resume.pdf")
        
        if resume_text.strip():
            # Generate recommendations
            recommended_jobs = recommender.recommend_jobs(resume_text)
            
            st.subheader("Job Recommendations")
            
            for i, job in enumerate(recommended_jobs, 1):
                with st.expander(f"{i}. {job['title']}"):
                    st.write(f"**Job ID:** {job['job_id']}")
                    st.write(f"**Similarity Score:** {job['similarity_score']:.2%}")
                    st.write(f"**Skills Required:** {', '.join(job['skills_required'])}")
                    st.write(f"**Experience Required:** {job.get('experience_required', {}).get('min_years', 'N/A')} years")
                    st.write(f"**Qualifications:** {job['qualifications']}")
                    
def resume_ranking():
    st.title("Resume Ranking System")

    st.header("Provide Job Description and Upload Resumes")

    # Input Job Description
    job_description_text = st.text_area("Paste Job Description", "")

    # Upload Resumes
    uploaded_resumes = st.file_uploader(
        "Upload Resumes (PDF or Text Files, Max 10)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Rank Resumes"):
        if job_description_text.strip() and uploaded_resumes:
            if len(uploaded_resumes) > 10:
                st.error("Please upload a maximum of 10 resumes.")
            else:
                try:
                    # Clean Job Description
                    job_description_cleaned = clean_text(job_description_text)

                    # Process Resumes
                    st.write("Processing resumes...")
                    resumes = []
                    filenames = []
                    for resume_file in uploaded_resumes:
                        if resume_file.name.endswith(".pdf"):
                            content = extract_text_from_pdf(resume_file)
                        elif resume_file.name.endswith(".txt"):
                            content = resume_file.read().decode("utf-8")
                        else:
                            st.error(f"Unsupported file type: {resume_file.name}")
                            continue

                        cleaned_content = clean_text(content)
                        resumes.append(cleaned_content)
                        filenames.append(resume_file.name)

                    # Load Sentence-BERT
                    st.write("Loading Sentence-BERT model...")
                    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

                    # Generate Embeddings
                    with ThreadPoolExecutor() as executor:
                        job_desc_embedding = embed_texts([job_description_cleaned], model)[0]
                        resume_embeddings = embed_texts(resumes, model)

                    # Normalize Embeddings
                    job_desc_embedding = job_desc_embedding / np.linalg.norm(job_desc_embedding)
                    resume_embeddings = normalize_vectors(resume_embeddings)

                    # Rank Resumes
                    ranked_resumes = rank_resumes(job_desc_embedding, resume_embeddings, filenames, resumes)

                    # Display Ranked Resumes
                    st.write("Ranked Resumes Based on Similarity Scores:")
                    for resume in ranked_resumes:
                        st.write(f"### Rank {resume['rank']}")
                        st.write(f"**File Name:** {resume['file_name']}")
                        st.write(f"**Similarity Score:** {resume['similarity_score']:.2f}%")
                        st.write(f"**Resume Content Preview:** {resume['content_preview']}...")
                        st.write("---")

                    # Download Option
                    st.write("Download Results")
                    results_df = pd.DataFrame(ranked_resumes)
                    results_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=results_csv,
                        file_name="ranked_resumes.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please provide a job description and upload at least one resume.")

def clean_text(text):
    return " ".join(text.split()).strip()

def embed_texts(texts, model):
    return model.encode(texts, batch_size=8, show_progress_bar=True)

def normalize_vectors(vectors):
    return np.array([v / np.linalg.norm(v) for v in vectors])


def visualization():
    # Streamlit UI
    st.title("Resume and Job Description Analyzer")

    # Resume Upload Section
    st.header("Upload Resume")
    resume_file = st.file_uploader("Upload your resume (PDF only)", type="pdf")

    # Job Description Text Area
    st.header("Enter Job Description")
    job_description_text = st.text_area("Paste job description text here", "")

    # Visualization Dropdown
    st.header("Select Visualization")
    visualization = st.selectbox(
        "Choose a visualization:",
        ["Word Cloud", "Venn", "Bar Chart", "Heatmap"]
    )

    
    if resume_file and job_description_text.strip():
        resume_text = extract_text_from_pdf(resume_file)
            
            
        documents = [resume_text, job_description_text]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
            
            
        if visualization == "Word Cloud":
            st.subheader("Word Clouds")
            generate_word_cloud(tfidf_matrix, feature_names, 0, "Resume Word Cloud")
            generate_word_cloud(tfidf_matrix, feature_names, 1, "Job Description Word Cloud")
        elif visualization == "Venn":
            st.subheader("Sankey Diagram")
            generate_sankey(tfidf_matrix, feature_names)
        elif visualization == "Bar Chart":
            st.subheader("Bar Chart")
            generate_bar_chart(tfidf_matrix, feature_names)
        elif visualization == "Heatmap":
            st.subheader("Heatmap")
            generate_heatmap(tfidf_matrix, feature_names)
        else:
            st.error("Please upload a resume and enter job description text.")



# Load the .env file
load_dotenv()

# Access environment variables
AWS_ACCESS_KEY_ID = "Enter your own key"
AWS_ACCESS_KEY = "Enter your own key"
AWS_SESSION_TOKEN = "Enter your own key"
REGION_NAME = "Enter your own Region"

def initialize_analyzer():
    return BedrockResumeAnalyzer(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_access_key=AWS_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=REGION_NAME,
    )

analyzer = initialize_analyzer()

def ResumeGFScore():
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "trigger_rerun" not in st.session_state:
        st.session_state["trigger_rerun"] = False

    # Page heading
    st.markdown("<h1 style='text-align: center; margin-top: 2em;'>Grammar and Format Check</h1>", unsafe_allow_html=True)

    # Display chat messages
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(
                f'<div class="message-container user-bubble"><div class="user-message">{content}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="message-container bot-bubble"><div class="bot-message">{content}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # CSS styling
    st.markdown("""
        <style>

        .user-message, .bot-message {
            background-color: #E2E2E2;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }

        .user-message {
            background-color: #DCF8C6;
        }

        .bot-message {
            background-color: #E2E2E2;
        }

        .chat-input-row {
            max-width: 800px;
            width: 100%;
            margin: 1em auto 0 auto;
            display: flex;
            align-items: center;
            background-color: #EFEFEF;
            border-radius: 30px;
            padding: 0em 1em;
            gap: 0.5em;
            box-sizing: border-box;
        }

        .chat-text-input {
            flex: 1;
            border: none;
            outline: none;
            padding: 0.5em;
            font-size: 16px;
            background-color: transparent;
        }

        .chat-text-input::placeholder {
            color: #C0C0C0;
        }

        .chat-submit-btn {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 0.5em;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 40px;
            width: 40px;
            flex-shrink: 0;
        }

        .chat-submit-btn:hover {
            background-color: #0056b3;
        }
        </style>
        """, unsafe_allow_html=True)

    # Chat input row
    st.markdown('<div class="chat-input-row">', unsafe_allow_html=True)

    # File uploader for PDF resumes (provide a label and then collapse it)
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type="pdf",
        label_visibility="collapsed",
        key="chat_file"
    )

    # Text input for user's message (provide a label and then collapse it)
    user_input = st.text_input(
        "User Message",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        key="chat_input",
        help="Type your message and press the send button"
    )

    # Send button
    send_clicked = st.button("➤", key="send_button", help="Send message")

    st.markdown('</div>', unsafe_allow_html=True)

    if send_clicked:
        user_message = user_input.strip()
        file_provided = (uploaded_file is not None)

        # If a file is provided, analyze it and append the result
        if file_provided:
            with st.spinner("Processing your resume..."):
                with open("uploaded_resume.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                resume_text = analyzer.extract_text_from_pdf("uploaded_resume.pdf")
                analysis_results = analyzer.analyze_resume_text(resume_text)

    #             analysis_summary = f"""
    # Grammar Score: {analysis_results.get('grammar_score', 'N/A')}/100  
    # Formatting Score: {analysis_results.get('formatting_score', 'N/A')}/100

    # Grammatical Errors:  
    # {', '.join([err['description'] for err in analysis_results.get('grammatical_errors', [])]) or 'None'}

    # Formatting Issues:  
    # {', '.join([issue['description'] for issue in analysis_results.get('formatting_issues', [])]) or 'None'}

    # Recommendations:  
    # {', '.join([rec['description'] for rec in analysis_results.get('recommendations', [])]) or 'None'}
    # """
    #             st.session_state["messages"].append({"role": "assistant", "content": analysis_summary})

    #     # Only if no file is provided, use the user_message logic
    #     elif user_message:
    #         st.session_state["messages"].append({"role": "user", "content": user_message})
    #         mock_response = f"You said: {user_message}. (This is a placeholder response.)"
    #         st.session_state["messages"].append({"role": "assistant", "content": mock_response})

    #     # If neither file nor message is provided, show an error
    #     if not file_provided and not user_message:
    #         st.error("Please upload a PDF or type a message.")
                analysis_summary = f"""
            Grammar Score: {analysis_results.get('grammar_score', 'N/A')}/100  
            Formatting Score: {analysis_results.get('formatting_score', 'N/A')}/100

            Grammatical Errors:  
            {', '.join([err['description'] for err in analysis_results.get('grammatical_errors', [])]) or 'None'}

            Formatting Issues:  
            {', '.join([issue['description'] for issue in analysis_results.get('formatting_issues', [])]) or 'None'}

            Recommendations:  
            {', '.join([rec['description'] for rec in analysis_results.get('recommendations', [])]) or 'None'}
            """
                st.text_area("Analysis Summary", analysis_summary, height=300)

        # Only if no file is provided, use the user_message logic
        elif user_message:
            st.text_area("User Message", user_message, height=100)
            mock_response = f"You said: {user_message}. (This is a placeholder response.)"
            st.text_area("Response", mock_response, height=100)

        # If neither file nor message is provided, show an error
        if not file_provided and not user_message:
            st.error("Please upload a PDF or type a message.")
    

    # Now display messages after all logic is done
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(
                f'<div class="message-container user-bubble"><div class="user-message">{content}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="message-container bot-bubble"><div class="bot-message">{content}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["Main Page", "Job Recommedation", "Ranking Resume","ResumeGFScore","Resume Generation Preview"])

with tabs[0]:
    main_page()
with tabs[1]:
    job_recommendation()
with tabs[2]:
    resume_ranking()
with tabs[3]:
    ResumeGFScore()
with tabs[4]:
    resume_generation_page()
