import os
import re
import nltk
import spacy


nlp = spacy.load("en_core_web_sm")


nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords


class ResumePreprocessor:
    def __init__(self, resume_folder, label_folder):
        self.resume_folder = resume_folder
        self.label_folder = label_folder
        self.stop_words = set(stopwords.words("english"))

    def load_resume(self, resume_path):
        with open(resume_path, encoding="utf8", errors='ignore') as file:
            text = file.read()
        return text

    def clean_text(self, text):
        # Remove extra whitespace and special characters
        text = re.sub('httpS+s*', ' ', text)  # remove URLs
        text = re.sub('RT|cc', ' ',text)  # remove RT and cc
        text = re.sub('#S+', '', text)  # remove hashtags
        text = re.sub('@S+', '  ', text)  # remove mentions
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ',text)  # remove punctuations
        text = re.sub(r'[^x00-x7f]', r' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9,.]', ' ', text)
        return text

    def preprocess_text(self, text):
        # Tokenize, remove stopwords, and lemmatize text
        doc = nlp(text)
        tokens = []

        for token in doc:
            if token.text.lower() not in self.stop_words and not token.is_punct:
                
                tokens.append(token.lemma_)


        return " ".join(tokens)

    def extract_sections(self, text):
        # Tokenize sentences
        doc = nlp(text)
        sections = {"Experience": [], "Education": [], "Skills": []}

        for sent in doc.sents:
            sentence = sent.text.strip()
            if re.search(r"\bExperience\b", sentence, re.IGNORECASE):
                sections["Experience"].append(sentence)
            elif re.search(r"\bEducation\b", sentence, re.IGNORECASE):
                sections["Education"].append(sentence)
            elif re.search(r"\bSkills?\b", sentence, re.IGNORECASE):
                sections["Skills"].append(sentence)

        # Clean and preprocess each section
        for section in sections:
            sections[section] = self.preprocess_text(" ".join(sections[section]))

        return sections

    def preprocess_all_resumes(self):
        resume_data = {}
        for filename in os.listdir(self.resume_folder):
            if filename.endswith(".txt"):
                resume_path = os.path.join(self.resume_folder, filename)
                resume_text = self.load_resume(resume_path)

                # Clean raw text
                cleaned_text = self.clean_text(resume_text)

                # Extract and preprocess sections
                sections = self.extract_sections(cleaned_text)

                resume_data[filename] = sections

        return resume_data



resume_folder = 'resumes_corpus'
label_folder = 'resumes_corpus'


preprocessor = ResumePreprocessor(resume_folder, label_folder)
processed_resumes = preprocessor.preprocess_all_resumes()


for filename, sections in processed_resumes.items():
    print(f"\nFilename: {filename}")
    print(f"Sections: {sections}")
    break
