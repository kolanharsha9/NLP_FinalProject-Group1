#%%
import re
from typing import Counter
from datasets import load_dataset

# %%
import re
import random
from datasets import load_dataset

def extract_sections(text, headers):
    sections = {}
    header_pattern = r'(' + '|'.join(re.escape(header) for header in headers) + r')'
    parts = re.split(header_pattern, text)
    current_section = None
    for part in parts:
        part = part.strip()
        if part in headers:  
            current_section = part
            sections[current_section] = []  
        elif current_section:  
            sections[current_section].append(part)
    sections = {header: ' '.join(content) for header, content in sections.items()}
    return sections

def highlight_sections(sections):
    highlighted_text = ""
    for header, content in sections.items():
        highlighted_text += f"<section>{header}</section>\n"
        highlighted_text += content + "\n\n"
    return highlighted_text.strip()

def preprocess_resume(text, headers):
    sections = extract_sections(text, headers)
    highlighted_text = highlight_sections(sections)
    return highlighted_text

def preprocess_random_samples(dataset, headers, num_samples=100):
    all_texts = [row['Resume_test'] for row in dataset['train'] if row['Resume_test'] is not None]
    random_samples = random.sample(all_texts, min(num_samples, len(all_texts)))
    preprocessed_samples = [preprocess_resume(text, headers) for text in random_samples]
    return preprocessed_samples

# got all these possible headers from the below code, can add if any missed
section_headers = [
    "Professional Summary", "Summary","SKILL SET", "Objective", "Experience", "Work History",
    "Education", "Hobbies","SOFTWARE SKILLS","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
    "Affiliations","TECHNICALSKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
]

dataset = load_dataset("InferencePrince555/Resume-Dataset")

preprocessed_lines = preprocess_random_samples(dataset, section_headers, num_samples=100)

def preprocess_first_n_samples(dataset, headers, n=100):
    # Combine all rows into a single list of texts, using the correct field
    all_texts = [row['Resume_test'] for row in dataset['train'] if row['Resume_test'] is not None]
    # Select the first n samples
    first_n_samples = all_texts[:n]
    # Preprocess each sample
    preprocessed_samples = [preprocess_resume(text, headers) for text in first_n_samples]
    return preprocessed_samples

preprocessed_first_100 = preprocess_first_n_samples(dataset, section_headers, n=100)

for idx, preprocessed in enumerate(preprocessed_first_100, start=1):
    print(f"Preprocessed Resume {idx}:\n")
    print(preprocessed)
    print("\n" + "="*80 + "\n")
# 
# for idx, preprocessed in enumerate(preprocessed_lines, start=1):
#     print(f"Preprocessed Resume {idx}:\n")
#     print(preprocessed)
#     print("\n" + "="*80 + "\n")




# %%
# getting all possible headers with regex
dataset = load_dataset("InferencePrince555/Resume-Dataset")

def extract_possible_headers(resume_text):
    if not isinstance(resume_text, str):  
        return []
    
    header_pattern = r"(?m)^(?:[A-Z][a-zA-Z\s]+|[A-Z\s]+)\:?"
    matches = re.findall(header_pattern, resume_text)
    return list(set(match.strip() for match in matches if len(match.strip()) > 2))

all_headers = []
for resume in dataset["train"]:
    resume_text = resume.get("Resume_test")  
    if resume_text:  
        headers = extract_possible_headers(resume_text)
        all_headers.extend(headers)

header_counts = Counter(all_headers)

for header, count in header_counts.most_common():
    print(f"{header}: {count}")
# %%
