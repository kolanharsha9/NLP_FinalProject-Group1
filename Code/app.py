import streamlit as st
import pandas as pd


st.title("Resume and Job Matcher")

st.sidebar.title("Menu")
menu_options = ["Home", "About"]
choice = st.sidebar.selectbox("Select an option", menu_options)

if choice == "Home":
    st.header("Job Description")
    job_description = st.text_area("Paste the job description here")

    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Drag and drop your resume here", type=["pdf", "docx"])

    if uploaded_file is not None:
        st.write("Uploaded Resume:", uploaded_file.name)
        st.header("Output Results")
        st.write("Results will be displayed here after processing.")

elif choice == "About":
    st.header("About")
    st.write("This is a basic Streamlit application for matching resumes with job descriptions.")


import re
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from PyPDF2 import PdfReader
import streamlit as st
from tqdm import tqdm

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Function to get BERT embedding for a text
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to compute similarity score
def compute_similarity(job_desc_text, resume_text, tokenizer, model):
    # Get embeddings for job description and resume
    job_desc_embedding = get_bert_embedding(job_desc_text, tokenizer, model)
    resume_embedding = get_bert_embedding(resume_text, tokenizer, model)

    # Compute similarity using cosine similarity
    similarity = np.dot(job_desc_embedding, resume_embedding) / (
        np.linalg.norm(job_desc_embedding) * np.linalg.norm(resume_embedding)
    )

    return similarity

# Integration in the middle of a larger Streamlit script
st.header('Checking Resume score')
st.header("Upload Files")
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Compute Similarity"):
    if job_desc_file is not None and resume_file is not None:
        # Load BERT tokenizer and model
        st.write("Loading BERT model...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Extract and preprocess text
        st.write("Processing files...")
        job_desc_text = clean_text(extract_text_from_pdf(job_desc_file))
        resume_text = clean_text(extract_text_from_pdf(resume_file))

        # Compute similarity score
        st.write("Computing similarity...")
        similarity_score = compute_similarity(job_desc_text, resume_text, tokenizer, model)

        # Display result
        st.success(f"Similarity Score: {similarity_score:.4f}")
    else:
        st.error("Please upload both the job description and the resume.")





# Helper functions
def embed_texts(texts, model, tokenizer):
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        chunks = []
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = tokens['input_ids']
        for i in range(0, input_ids.size(1), 512):
            chunk = {key: val[:, i:i+512] for key, val in tokens.items()}
            with torch.no_grad():
                output = model(**chunk)
            chunks.append(output.last_hidden_state[:, 0, :].mean(0).numpy())
        embeddings.append(np.mean(chunks, axis=0))
    return np.array(embeddings)



def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Streamlit App
st.header("Resume Ranking System")
st.subheader("Upload Job Description and Resumes")

# Upload Job Description
uploaded_pdf = st.file_uploader("Upload Job Description (PDF)", type="pdf")

# Upload Resumes
uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF or Text Files, Max 10)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Parameters
if st.button("Rank Resumes"):
    if uploaded_pdf and uploaded_resumes:
        if len(uploaded_resumes) > 10:
            st.error("Please upload a maximum of 10 resumes.")
        else:
            # Extract Job Description
            job_description = extract_text_from_pdf(uploaded_pdf)

            # Load Resumes
            st.write("Processing resumes...")
            resumes = []
            filenames = []
            for resume_file in uploaded_resumes:
                try:
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
                except Exception as e:
                    st.error(f"Error processing {resume_file.name}: {str(e)}")

            # Load Pretrained BERT
            st.write("Loading BERT model...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")

            # Embed Resumes and Job Description
            st.write("Generating embeddings for resumes...")
            resume_embeddings = embed_texts(resumes, model, tokenizer)
            st.write("Generating embedding for job description...")
            job_desc_embedding = embed_texts([job_description], model, tokenizer)[0]

            # Normalize Embeddings
            resume_embeddings = normalize_vectors(resume_embeddings)
            job_desc_embedding = job_desc_embedding / np.linalg.norm(job_desc_embedding)

            # Calculate Similarity Scores
            similarity_scores = np.dot(resume_embeddings, job_desc_embedding)

            # Rank Resumes by Similarity Scores
            ranked_indices = np.argsort(similarity_scores)[::-1]

            # Display Ranked Resumes
            st.write("Ranked Resumes Based on Similarity Scores:")
            for rank, idx in enumerate(ranked_indices, start=1):
                st.write(f"### Rank {rank}")
                st.write(f"**File Name:** {filenames[idx]}")
                st.write(f"**Similarity Score:** {similarity_scores[idx] * 100:.2f}%")
                st.write(f"**Resume Content:** {resumes[idx][:500]}...")
                st.write("---")
    else:
        st.error("Please upload a job description PDF and at least one resume.")





#%%



#gives you json for resume and jobdes

import streamlit as st
from resume_job_description_parser import process_documents, save_to_json

def main():
    # Streamlit UI components
    st.title("Resume and Job Description Parser")
    
    # API Key input
    api_key = st.text_input("Enter Gemini API Key")
    
    # File uploaders
    resume_file = st.file_uploader("Upload Resume PDF", type=['pdf'])
    job_desc_file = st.file_uploader("Upload Job Description", type=['txt'])
    
    if st.button("Parse Documents"):
        # Save uploaded files temporarily
        with open("temp_resume.pdf", "wb") as f:
            f.write(resume_file.getbuffer())
        
        with open("temp_job_desc.txt", "wb") as f:
            f.write(job_desc_file.getbuffer())
        
        # Process documents
        parsed_resume, parsed_job_description = process_documents(
            api_key, 
            "temp_resume.pdf", 
            "temp_job_desc.txt"
        )
        
        # Display or save results
        if parsed_resume:
            st.json(parsed_resume)
            save_to_json(parsed_resume, 'parsed_resume.json')
        
        if parsed_job_description:
            st.json(parsed_job_description)
            save_to_json(parsed_job_description, 'parsed_job_description.json')

if __name__ == "__main__":
    main()



# %%
import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from resume_job_description_parser import process_documents
from PyPDF2 import PdfReader
# Function to extract text from a PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to generate a word cloud based on TF-IDF
def generate_word_cloud(tfidf_matrix, feature_names, document_index, title):
    scores = tfidf_matrix[document_index].toarray().flatten()
    word_scores = {feature_names[i]: scores[i] for i in range(len(scores))}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

# Function to generate a Sankey diagram
def generate_sankey(tfidf_matrix, feature_names):
    # Extract TF-IDF scores for both documents
    resume_scores = tfidf_matrix[0].toarray().flatten()
    job_description_scores = tfidf_matrix[1].toarray().flatten()

    # Filter terms that appear in both documents
    common_terms = [
        (term, resume_scores[idx], job_description_scores[idx])
        for idx, term in enumerate(feature_names)
        if resume_scores[idx] > 0 and job_description_scores[idx] > 0
    ]

    # Prepare data for Sankey diagram
    source = []  # Resume
    target = []  # Job Description
    values = []  # Strength of connection

    for term, resume_score, job_desc_score in common_terms:
        source.append(f"Resume: {term}")
        target.append(f"Job Desc: {term}")
        values.append(min(resume_score, job_desc_score))  # Use minimum score as strength

    # Map unique labels to indices
    all_labels = list(set(source + target))
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    # Map data to indices for Sankey diagram
    sankey_source = [label_to_idx[label] for label in source]
    sankey_target = [label_to_idx[label] for label in target]

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels
        ),
        link=dict(
            source=sankey_source,
            target=sankey_target,
            value=values
        )
    )])

    fig.update_layout(title_text="Semantic Flow of Key Terms Between Resume and Job Description", font_size=10)

    # Display the Sankey diagram
    st.plotly_chart(fig)

# Function to generate a bar chart comparing TF-IDF scores
def generate_bar_chart(tfidf_matrix, feature_names):
    # Extract TF-IDF scores for both documents
    resume_scores = tfidf_matrix[0].toarray().flatten()
    job_description_scores = tfidf_matrix[1].toarray().flatten()

    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Word': feature_names,
        'Resume': resume_scores,
        'Job Description': job_description_scores
    })

    # Sort by importance in the job description
    df = df.sort_values(by='Job Description', ascending=False)

    # Plotting the bar chart
    x = np.arange(len(df['Word']))  # Word indices
    width = 0.4  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    # Resume bar
    ax.bar(x - width / 2, df['Resume'], width, label='Resume', color='blue')

    # Job Description bar
    ax.bar(x + width / 2, df['Job Description'], width, label='Job Description', color='green')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df['Word'], rotation=45, ha='right')
    ax.set_xlabel("Top Words", fontsize=12)
    ax.set_ylabel("TF-IDF Scores", fontsize=12)
    ax.set_title("Word Importance Comparison: Resume vs Job Description", fontsize=14)
    ax.legend()

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Function to generate a similarity heatmap
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity
def generate_heatmap(resume_text, job_description_text):
    api_key = "AIzaSyDw1PTBcbK09IYvQkUI7Fp39A8M1NMm-Pg"

    # Process documents using the Gemini API
    parsed_resume, parsed_job_description = process_documents(api_key, resume_file, job_description_file)

    if parsed_resume is None or parsed_job_description is None:
        st.error("Failed to parse the documents. Please check the files and try again.")
    else:
        # Extract skills and experiences from the parsed JSON
        resume_skills = " ".join(parsed_resume.get("skills", []))
        st.write(resume_skills)
        job_description_skills = " ".join(parsed_job_description.get("required_skills", []))
        st.write(job_description_skills)

        resume_experience = " ".join(
            [
                " ".join(exp.get("responsibilities", []))
                for exp in parsed_resume.get("work_experience", [])
            ]
        )
        st.write(resume_experience)
        job_description_experience = " ".join(
            parsed_job_description.get("responsibilities", [])
            + parsed_job_description.get("qualifications", [])
            + parsed_job_description.get("preferred_qualifications", [])
        )
        st.write(job_description_experience)

        # Calculate similarity scores for skills and experience
        similarity_scores = {
            "Skills": calculate_similarity(resume_skills, job_description_skills),
            "Experience": calculate_similarity(resume_experience, job_description_experience),
        }

        # Prepare data for heatmap
        heatmap_data = [list(similarity_scores.values())]
        heatmap_labels = list(similarity_scores.keys())

        # Plot the heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="YlGnBu",
            xticklabels=heatmap_labels,
            yticklabels=["Similarity"],
        )
        plt.title("Similarity Scores for Resume and Job Description")
        plt.xlabel("Sections")
        plt.ylabel("")
        plt.tight_layout()

        # Display the heatmap in Streamlit
        st.pyplot(plt)

# Streamlit app
st.header("Resume and Job Description Analysis")
# Upload files
resume_file = st.file_uploader("Upload Resume PDF", type="pdf")
job_description_file = st.file_uploader("Upload Job Description PDF", type="pdf")

# Check if both files are uploaded
if resume_file and job_description_file:
    # Extract text from uploaded files
    try:
        resume_text = extract_text_from_pdf(resume_file)
        job_description_text = extract_text_from_pdf(job_description_file)

        # Generate TF-IDF matrix
        documents = [resume_text, job_description_text]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Graph type selection
        graph_type = st.selectbox(
            "Select the type of graph:",
            ["Word Cloud", "Sankey Diagram", "Bar Chart", "Similarity Heatmap"],
        )

        # Generate the selected graph
        if graph_type == "Word Cloud":
            generate_word_cloud(tfidf_matrix, feature_names, 0, "Resume Word Cloud")
            generate_word_cloud(tfidf_matrix, feature_names, 1, "Job Description Word Cloud")
        elif graph_type == "Sankey Diagram":
            generate_sankey(tfidf_matrix, feature_names)
        elif graph_type == "Bar Chart":
            generate_bar_chart(tfidf_matrix, feature_names)
        elif graph_type == "Similarity Heatmap":
            generate_heatmap(resume_text, job_description_text)

    except Exception as e:
        st.error(f"An error occurred while processing the files: {str(e)}")
else:
    st.info("Please upload both a resume and a job description to proceed.")


