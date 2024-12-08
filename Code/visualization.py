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
import streamlit as st
# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
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
    
    # Render in Streamlit
    st.pyplot(fig)


# Function to generate a Sankey diagram
import plotly.graph_objects as go

def generate_sankey(tfidf_matrix, feature_names):
    resume_scores = tfidf_matrix[0].toarray().flatten()
    job_description_scores = tfidf_matrix[1].toarray().flatten()

    common_terms = [
        (term, resume_scores[idx], job_description_scores[idx])
        for idx, term in enumerate(feature_names)
        if resume_scores[idx] > 0 and job_description_scores[idx] > 0
    ]

    source, target, values = [], [], []

    for term, resume_score, job_desc_score in common_terms:
        source.append(f"Resume: {term}")
        target.append(f"Job Desc: {term}")
        values.append(min(resume_score, job_desc_score))

    all_labels = list(set(source + target))
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    sankey_source = [label_to_idx[label] for label in source]
    sankey_target = [label_to_idx[label] for label in target]

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_labels),
        link=dict(source=sankey_source, target=sankey_target, value=values)
    )])

    fig.update_layout(title_text="Semantic Flow of Key Terms Between Resume and Job Description", font_size=10)
    
    # Render in Streamlit
    st.plotly_chart(fig)


# Function to generate a bar chart comparing TF-IDF scores
import pandas as pd
import numpy as np

def generate_bar_chart(tfidf_matrix, feature_names):
    resume_scores = tfidf_matrix[0].toarray().flatten()
    job_description_scores = tfidf_matrix[1].toarray().flatten()

    df = pd.DataFrame({
        'Word': feature_names,
        'Resume': resume_scores,
        'Job Description': job_description_scores
    }).sort_values(by='Job Description', ascending=False)

    x = np.arange(len(df['Word']))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df['Resume'], width, label='Resume', color='blue')
    ax.bar(x + width / 2, df['Job Description'], width, label='Job Description', color='green')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Word'], rotation=45, ha='right')
    ax.set_xlabel("Top Words")
    ax.set_ylabel("TF-IDF Scores")
    ax.set_title("Word Importance Comparison: Resume vs Job Description")
    ax.legend()
    plt.tight_layout()
    
    # Render in Streamlit
    st.pyplot(fig)


import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def generate_heatmap(parsed_resume, parsed_job_description):
    resume_skills = " ".join(parsed_resume.get("skills", []))
    job_description_skills = " ".join(parsed_job_description.get("required_skills", []))

    resume_experience = " ".join([
        " ".join(exp.get("responsibilities", []))
        for exp in parsed_resume.get("work_experience", [])
    ])
    job_description_experience = " ".join(
        parsed_job_description.get("responsibilities", []) +
        parsed_job_description.get("qualifications", []) +
        parsed_job_description.get("preferred_qualifications", [])
    )

    similarity_scores = {
        "Skills": calculate_similarity(resume_skills, job_description_skills),
        "Experience": calculate_similarity(resume_experience, job_description_experience),
    }

    heatmap_data = [list(similarity_scores.values())]
    heatmap_labels = list(similarity_scores.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        heatmap_data, annot=True, cmap="YlGnBu",
        xticklabels=heatmap_labels, yticklabels=["Similarity"], ax=ax
    )
    ax.set_title("Similarity Scores for Resume and Job Description")
    ax.set_xlabel("Sections")
    plt.tight_layout()
    
    # Render in Streamlit
    st.pyplot(fig)


# Main script
if __name__ == "__main__":
    resume_file_path = 'processed_resume/HarshavardanaReddyKolan_Resume.pdf'
    job_description_file_path = 'Processed_jd/Job-Description_DataScientist.pdf'


    resume_text = extract_text_from_pdf(resume_file_path)
    job_description_text = extract_text_from_pdf(job_description_file_path)

    documents = [resume_text, job_description_text]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    generate_word_cloud(tfidf_matrix, feature_names, 0, "Resume Word Cloud")
    generate_word_cloud(tfidf_matrix, feature_names, 1, "Job Description Word Cloud")

    generate_sankey(tfidf_matrix, feature_names)
    generate_bar_chart(tfidf_matrix, feature_names)

    api_key = "AIzaSyDw1PTBcbK09IYvQkUI7Fp39A8M1NMm-Pg"
    parsed_resume, parsed_job_description = process_documents(api_key, resume_file_path, job_description_file_path)
    generate_heatmap(parsed_resume, parsed_job_description)

