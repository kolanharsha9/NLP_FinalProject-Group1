#%%
# import re
# from typing import Counter
# from datasets import load_dataset

# %%
# import re
# import random
# from datasets import load_dataset

# def extract_sections(text, headers):
#     sections = {}
#     header_pattern = r'(' + '|'.join(re.escape(header) for header in headers) + r')'
#     parts = re.split(header_pattern, text)
#     current_section = None
#     for part in parts:
#         part = part.strip()
#         if part in headers:  
#             current_section = part
#             sections[current_section] = []  
#         elif current_section:  
#             sections[current_section].append(part)
#     sections = {header: ' '.join(content) for header, content in sections.items()}
#     return sections

# def highlight_sections(sections):
#     highlighted_text = ""
#     for header, content in sections.items():
#         highlighted_text += f"<section>{header}</section>\n"
#         highlighted_text += content + "\n\n"
#     return highlighted_text.strip()

# def preprocess_resume(text, headers):
#     sections = extract_sections(text, headers)
#     highlighted_text = highlight_sections(sections)
#     return highlighted_text

# def preprocess_random_samples(dataset, headers, num_samples=100):
#     all_texts = [row['Resume_test'] for row in dataset['train'] if row['Resume_test'] is not None]
#     random_samples = random.sample(all_texts, min(num_samples, len(all_texts)))
#     preprocessed_samples = [preprocess_resume(text, headers) for text in random_samples]
#     return preprocessed_samples

# # got all these possible headers from the below code, can add if any missed
# section_headers = [
#     "Professional Summary", "Summary","SKILL SET", "Objective", "Experience", "Work History",
#     "Education", "Hobbies","SOFTWARE SKILLS","Technical Skill Details","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
#     "Affiliations","Company Details","TECHNICALSKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
# ]
# #%%
# dataset = load_dataset("InferencePrince555/Resume-Dataset")

# preprocessed_lines = preprocess_random_samples(dataset, section_headers, num_samples=100)

# def preprocess_first_n_samples(dataset, headers, n=100):
#     # Combine all rows into a single list of texts, using the correct field
#     all_texts = [row['Resume_test'] for row in dataset['train'] if row['Resume_test'] is not None]
#     # Select the first n samples
#     first_n_samples = all_texts[:n]
#     # Preprocess each sample
#     preprocessed_samples = [preprocess_resume(text, headers) for text in first_n_samples]
#     return preprocessed_samples

# preprocessed_first_100 = preprocess_first_n_samples(dataset, section_headers, n=100)

# for idx, preprocessed in enumerate(preprocessed_first_100, start=1):
#     print(f"Preprocessed Resume {idx}:\n")
#     print(preprocessed)
#     print("\n" + "="*80 + "\n")
# # 
# # for idx, preprocessed in enumerate(preprocessed_lines, start=1):
# #     print(f"Preprocessed Resume {idx}:\n")
# #     print(preprocessed)
# #     print("\n" + "="*80 + "\n")
# #%%
# text="ACCOUNTANT Summary Seasoned professional accounting professional skilled at managing multiple accounts across diverse financial and HR systems An analytical leader with outstanding leadership qualities excellent communication skills and an attention to details Experience Accountant 04 2015 to Current Company Name City State Responsible for the management of multi departmental account reconciliations including cash receipts general Ledger reports as well as reconciling vendor invoices Generated weekly monthly semi annual budget based on company s sales projections Analyzed actuals against revenue generated from various departments such as payroll tax payables medical insurance etc Ensuring compliance with internal Sarbanes Oxley SOX 404 procedures Performed audit activities involving auditing audits bank reconciliation income statements and treasury functions in accordance with generally accepted accounting practices Reviewed Bank Reconciliation Statement EAR P L Accounts Payable Reports Monitored General Ledger Fixed Asset Accounting which includes entering data into a system Assisted Controller accountant by preparing work schedules ensuring proper documentation is being maintained Served as primary contact for client regarding issues related to payment status Prepared expense forecasts that include variance variances adjusting gross margin while maintaining accuracy Key Accomplishments Manage all year end operating expenses within established guidelines Schedule month end closing reviews Maintained over 150 000 project payments annually Collaborated extensively with Finance Department team members during special projects Successfully managed 6 separate acquisitions simultaneously through successful divestiture Oversaw new employee orientation training program consisting of 25 employees Education 2013 BACHELORS OF BUSINESS OPERATION APPLIED SCIENCE Business Administration University of Puerto Rico City State Certifications FINANCIAL AND PERSONAL ACCOUNTANT August 2011 to May 2014 Company Name City State Gathered analyzed and resolved key operational issues Developed detailed business processes Evaluated cost reduction opportunities Worked effectively with upper management to define requirements Sustained positive relations with clients vendors and external partners Established clear expectations about operations and future goals Kept track of tasks completed throughout entire tenure Handled escalated situations when necessary Provided exceptional customer service Resolved complex technical issues accurately and expediently Actively pursued solutions provided exemplary customer service Negotiated pricing agreements contracts fees and other third party products Processed Purchase Orders via QuickBooks Salesforce Confidentiality Verification Report AFRA Bookkeeper November 2010 to February 2011 Company Name City State Operated electronic filing system Took charge of day to day administrative duties Conducted quality assurance tests Implemented corrective care methods Answered phone calls wrote correspondence Completed daily inventory counts Wrote automated test cases Executed adhoc reporting Used statistical analysis to develop marketing strategies Communicated product changes information risks and trends to appropriate stakeholders Tr"
# cleaned=preprocess_resume(text,section_headers)
# print(cleaned)
# # %%
# # getting all possible headers with regex
# dataset = load_dataset("InferencePrince555/Resume-Dataset")

# def extract_possible_headers(resume_text):
#     if not isinstance(resume_text, str):  
#         return []
    
#     header_pattern = r"(?m)^(?:[A-Z][a-zA-Z\s]+|[A-Z\s]+)\:?"
#     matches = re.findall(header_pattern, resume_text)
#     return list(set(match.strip() for match in matches if len(match.strip()) > 2))

# all_headers = []
# for resume in dataset["train"]:
#     resume_text = resume.get("Resume_test")  
#     if resume_text:  
#         headers = extract_possible_headers(resume_text)
#         all_headers.extend(headers)

# header_counts = Counter(all_headers)

# for header, count in header_counts.most_common():
#     print(f"{header}: {count}")

#####################################################################################################################################
#  The above code was for testing and the below function is the one that is being integrated with stream app for resume genration
#####################################################################################################################################

# %%

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

class gen_resume: 
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # model_path = "models_bart1"  
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate_resume(self, instruction, model_path, max_length=1024):
        """Generate a resume given an instruction."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        inputs = tokenizer(
            f"generate_resume: {instruction}",
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        # inputs = tokenizer(
        #     f"generate_resume: {instruction}",
        #     return_tensors="pt",
        #     max_length=max_length,
        #     truncation=True,
        # )
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(
            inputs["input_ids"],
            max_length=1024,
            do_sample=True,
            top_k=50,                 
            top_p=0.9, 
            temperature=.7,            
            repetition_penalty=5.0,     
            num_beams=6,                
            no_repeat_ngram_size=5,     
            early_stopping=True         
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def clean_resume(self, text):
        lines = text.split("\n")
        seen = set()
        cleaned_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\s+", " ", text)
        sections = ["Professional Summary", "Work Experience", "Education", "Skills"]
        for section in sections:
            text = re.sub(section, f"\n{section.upper()}\n", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
        return text.strip()
    
    # def format_resume(self, text, sections):
    #     """Format the resume text based on section headers."""
    #     for section in sections:
    #         text = re.sub(rf"(?i)({section})\s*:", rf"\n{section.upper()}:", text)
    #     text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
    #     return text.strip()
    
    def format_resume(text, sections):
        """Format the resume text based on section headers."""
        # Capitalize and format sections for better readability
        # text = re.sub(r"(?i)([A-Z][A-Z\s]*):", r"\n\1\n", text)
        for section in sections:
            text = re.sub(rf"(?i)\b({section})\b:", rf"\n{section.upper()}:", text)
        # Add bullet points for lists
        text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
        
        return text.strip()
    
    def main(self, instruction):
        section_headers = [
    "Professional Summary", "Summary","SKILL SET", "Objective", "Experience", "Work History",
    "Education", "Hobbies","SOFTWARE SKILLS","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
    "Affiliations","TECHNICALSKILLS","TECHNICAL PROFICIENCIES","Additional Information SKILLS","SUMMARY OF SKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
]
    # instruction = "Generate a resume for a software engineer job."
        generated_resume = self.generate_resume(instruction, self.model, self.tokenizer)
        cleaned_resume = self.format_resume(generated_resume, section_headers)
        # print("Cleaned Resume:")
        # print(cleaned_resume)
        return cleaned_resume