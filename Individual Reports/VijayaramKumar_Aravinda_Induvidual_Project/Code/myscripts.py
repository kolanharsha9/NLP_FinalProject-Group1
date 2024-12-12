# ALL CODES ARE COMMENTED IN THIS FILE, TO RUN A CODE, PLEASE REFER TO THE FILE NAMES AND RUN IT FROM THE FILES IN THE MAIN CODE FOLDER IN THE REPOSITORY


#########################################
# model2.py
#########################################

# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import json
# from tensorflow.keras.preprocessing.text import Tokenizer
# from torch.nn.utils.rnn import pad_sequence
# #%%
# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #%%
# base_dir= os.path.abspath('../..')
# resume_dir = os.path.join(base_dir,'assests','data','resume_data.json')
# job_dir = os.path.join(base_dir,'assests','data','jobs_data.json')

# with open(job_dir, "r") as f:
#     job_descriptions = json.load(f)


# with open(resume_dir, "r") as f:
#     resumes = json.load(f)
# #%%
# # Extracting relevant features
# job_texts = [
#     " ".join([
#         " ".join(job["skills_required"]),
#         str(job["experience_required"]["min_years"]),
#         str(job["experience_required"]["max_years"]),
#         job["qualifications"],
#         job["responsibilities"]
#     ]) for job in job_descriptions
# ]

# resume_texts = [
#     " ".join([
#         " ".join(resume["skills"]["technical_skills"]),
#         " ".join(resume["experience"]["details"]),
#         " ".join(resume["education"]["details"]),
#         resume["responsibilities"],
#         resume["summary"]
#     ]) for resume in resumes
# ]
# #%%

# job_titles = [job["title"] for job in job_descriptions]
# label_encoder = LabelEncoder()
# job_labels = label_encoder.fit_transform(job_titles)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(job_texts + resume_texts)

# job_sequences = tokenizer.texts_to_sequences(job_texts)
# resume_sequences = tokenizer.texts_to_sequences(resume_texts)

# max_length = max(max(len(seq) for seq in job_sequences), max(len(seq) for seq in resume_sequences))
# job_padded = pad_sequence([torch.tensor(seq) for seq in job_sequences], batch_first=True, padding_value=0)
# resume_padded = pad_sequence([torch.tensor(seq) for seq in resume_sequences], batch_first=True, padding_value=0)
# #%%
# class JobDescriptionDataset(Dataset):
#     def __init__(self, text_data, labels):
#         self.text_data = text_data
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.text_data[idx], self.labels[idx]

# job_train, job_test, label_train, label_test = train_test_split(job_padded, job_labels, test_size=0.2, random_state=42)

# train_dataset = JobDescriptionDataset(job_train, label_train)
# test_dataset = JobDescriptionDataset(job_test, label_test)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# #%%
# # Model
# class JobDescriptionModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
#         super(JobDescriptionModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, num_classes)

#     def forward(self, input_text, return_hidden=False):
#         embedded = self.embedding(input_text)
#         _, (hidden, _) = self.lstm(embedded)
#         if return_hidden:
#             return hidden.squeeze(0)  
#         output = self.fc(hidden.squeeze(0)) 
#         return output
# #%%
# # Model parameters
# vocab_size = len(tokenizer.word_index) + 1
# embedding_dim = 64
# hidden_dim = 128
# num_classes = len(label_encoder.classes_)

# model = JobDescriptionModel(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# #%%
# # Training loop
# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for input_text, labels in train_loader:
#         input_text, labels = input_text.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(input_text)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# torch.save(model.state_dict(), "job_description_model.pth")
# print("Model saved!")
# #%%
# # Evaluation
# from sklearn.metrics import cohen_kappa_score
# model.eval()
# correct = 0
# total = 0
# all_labels = []
# all_predictions = []
# with torch.no_grad():
#     for input_text, labels in test_loader:
#         input_text, labels = input_text.to(device), labels.to(device)
#         outputs = model(input_text)
#         _, predicted = torch.max(outputs, dim=1)
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())
# kappa_score = cohen_kappa_score(all_labels, all_predictions)
# print(f"Test Accuracy: {correct / total:.4f}")
# print(f"Cohen's Kappa: {kappa_score:.4f}")
# #%%
# def get_embedding(model, text_sequence):
#     model.eval()
#     with torch.no_grad():
#         input_tensor = pad_sequence([torch.tensor(text_sequence)], batch_first=True, padding_value=0).to(device)
#         embedding = model(input_tensor, return_hidden=True)  
#     return embedding.cpu().numpy()

# from sklearn.metrics.pairwise import cosine_similarity

# resume_text_sequence = tokenizer.texts_to_sequences([" ".join(resume_texts[0])])[0]
# job_text_sequence = tokenizer.texts_to_sequences([" ".join(job_texts[0])])[0]

# resume_embedding = get_embedding(model, resume_text_sequence)
# job_embedding = get_embedding(model, job_text_sequence)

# similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
# print(f"Similarity Score: {similarity_score:.4f}")


# def provide_feedback(resume, job_description, tokenizer, model):

#     resume_sequence = tokenizer.texts_to_sequences([" ".join(resume)])[0]
#     job_sequence = tokenizer.texts_to_sequences([" ".join(job_description)])[0]
    
#     resume_embedding = get_embedding(model, resume_sequence)
#     job_embedding = get_embedding(model, job_sequence)
    

#     similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    
#     missing_skills = set(job_description["skills_required"]) - set(resume["skills"]["technical_skills"])
    
#     feedback = {
#         "similarity_score": similarity_score,
#         "missing_skills": list(missing_skills),
#         "recommendation": f"Consider adding the following skills: {', '.join(missing_skills)}"
#     }
#     return feedback

# feedback = provide_feedback(resumes[0], job_descriptions[0], tokenizer, model)
# print("Feedback:", feedback)

# #%%




# #%%

# from sklearn.metrics.pairwise import cosine_similarity

# def encode_text(model, text_input):
#     model.eval()
#     with torch.no_grad():
#         text_seq = tokenizer.texts_to_sequences([text_input])
#         text_tensor = pad_sequence([torch.tensor(text_seq[0])], batch_first=True, padding_value=0).to(device)
#         embedded = model.embedding(text_tensor)
#         _, (hidden, _) = model.lstm(embedded)
#         return hidden.squeeze(0).cpu().numpy()

# def compare_resume_to_job(resume_text, job_text):
#     resume_embedding = encode_text(model, resume_text)
#     job_embedding = encode_text(model, job_text)
#     resume_embedding_flat = resume_embedding.flatten()
#     job_embedding_flat = job_embedding.flatten()
#     similarity = cosine_similarity([resume_embedding_flat], [job_embedding_flat])[0][0]
#     return similarity
# #%%
# # Feedback function
# def provide_feedback(resume, job_description):
#     resume_text = " ".join([
#         " ".join(resume["skills"]["technical_skills"]),
#         " ".join(resume["experience"]["details"]),
#         " ".join(resume["education"]["details"]),
#         resume["responsibilities"],
#         resume["summary"]
#     ])
#     job_text = " ".join([
#         " ".join(job_description["skills_required"]),
#         str(job_description["experience_required"]["min_years"]),
#         str(job_description["experience_required"]["max_years"]),
#         job_description["qualifications"],
#         job_description["responsibilities"]
#     ])
    
#     similarity = compare_resume_to_job(resume_text, job_text)
#     missing_skills = set(job_description["skills_required"]) - set(resume["skills"]["technical_skills"])

#     feedback = {
#         "similarity_score": similarity,
#         "missing_skills": list(missing_skills)
#     }
#     return feedback


# resume = resumes[0]
# job = job_descriptions[0]
# feedback = provide_feedback(resume, job)
# print("Feedback:", feedback)


#########################################
# model3.py
#########################################
# #%% Import Libraries
# import nltk
# from nltk.corpus import stopwords
# import re
# from datasets import load_dataset
# from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
# from transformers import DataCollatorForSeq2Seq
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from collections import Counter
# from datasets import DatasetDict, Dataset

# #%% Step 1: Load Dataset
# dataset = load_dataset("InferencePrince555/Resume-Dataset")

# # Cleaning
# def clean_text(text):
#     if text is None:
#         return ""
#     text = re.sub(r"<.*?>", "", text)  # HTML tags
#     text = re.sub(r"span l.*?span", "", text)  # invalid spans
#     text = re.sub(r"\s+", " ", text.strip())  # extra whitespace
#     return text

# dataset = dataset.map(lambda x: {"Resume_test": clean_text(x["Resume_test"])})

# #%%

# # def resample_dataset(dataset, target_count):
# #     instruction_counts = Counter(dataset['train']['instruction'])
# #     instruction_datasets = {instruction: [] for instruction in instruction_counts}
# #     for example in dataset['train']:
# #         instruction_datasets[example['instruction']].append(example)
# #     resampled_data = []
# #     for examples in instruction_datasets.values():
# #         if len(examples) < target_count:
# #             resampled_data.extend(examples * (target_count // len(examples)) + examples[:target_count % len(examples)])
# #         else:
# #             resampled_data.extend(examples)
# #     resampled_dataset = Dataset.from_dict({key: [example[key] for example in resampled_data] for key in resampled_data[0]})
    
# #     return DatasetDict({'train': resampled_dataset})
# # target_count = 1500
# # dataset = resample_dataset(dataset, target_count)

# #%% Initializing model
# model_name = "facebook/bart-base"
# tokenizer = BartTokenizer.from_pretrained(model_name)

# #%% Preprocessing

# def extract_sections(text, headers):
#     sections = {}
#     header_pattern = r'\b(' + '|'.join(re.escape(header) for header in headers) + r')\b'
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
#         highlighted_text += f"{header.upper()}:\n"
#         highlighted_text += content + "\n\n"
#     return highlighted_text.strip()

# def preprocess_resume(text, headers):
#     sections = extract_sections(text, headers)
#     highlighted_text = highlight_sections(sections)
#     highlighted_text = re.sub(r'\s+', ' ', highlighted_text).strip()
#     return highlighted_text

# section_headers = [
#     "Professional Summary", "Summary","SKILL SET", "Objective", "Experience","EXPERIENCE", "Work History",
#     "Education", "Hobbies","SOFTWARE SKILLS","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
#     "Affiliations","TECHNICALSKILLS","TECHNICAL PROFICIENCIES","Additional Information SKILLS","SUMMARY OF SKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
# ]

# #%%

# def preprocess_dataset(dataset, headers):
#     def preprocess_row(row):
#         if row['Resume_test'] is None:
#             return {"Resume_test": ""}
#         return {"Resume_test": preprocess_resume(row['Resume_test'], headers)}
#     return dataset.map(preprocess_row)
# dataset = preprocess_dataset(dataset, section_headers)

# # first and last row of dataset

# # print("First row of dataset:")
# # print(dataset['train'][0])

# # print("Last row of dataset:")
# # print(dataset['train'][-1])
# #%%
# def preprocess_data(examples):

#     inputs = tokenizer(examples["instruction"], max_length=1024, truncation=True, padding="max_length")
#     outputs = tokenizer(examples["Resume_test"], max_length=1024, truncation=True, padding="max_length")
    
#     inputs["labels"] = outputs["input_ids"]
#     return inputs

# # Apply preprocessing and data split
# # tokenized_dataset = dataset["train"].map(preprocess_data, batched=True)
# train_test_split = dataset['train'].train_test_split(test_size=0.1)
# tokenized_train = train_test_split["train"].map(preprocess_data, batched=True)
# tokenized_test = train_test_split["test"].map(preprocess_data, batched=True)


# #%% Load Model
# model = BartForConditionalGeneration.from_pretrained(model_name)
# # Data collator for dynamic padding
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# #%% Training Arguments
# # training_args = TrainingArguments(
# #     output_dir="./resume_generator_bart",
# #     evaluation_strategy="epoch",
# #     learning_rate=5e-5,
# #     per_device_train_batch_size=8,
# #     per_device_eval_batch_size=8,
# #     num_train_epochs=5,
# #     save_strategy="epoch",
# #     logging_dir="./logs",
# #     logging_steps=50,
# #     save_total_limit=2
# # )


# training_args = TrainingArguments(
#     output_dir="./resume_generator_bart",
#     evaluation_strategy="steps",
#     eval_steps=500,
#     learning_rate=5e-4,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=20,
#     save_strategy="steps",
#     save_steps=500,
#     logging_dir="./logs",
#     logging_steps=50,
#     save_total_limit=2,
#     fp16=True,  
#     gradient_accumulation_steps=4  
# )



# # Initializing Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_test,
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# #%% Training the Model
# trainer.train()

# #%% Saving the Model
# trainer.save_model('../models_bart2')

# #%% Generate and and eval
# def generate_resume(instruction, model, tokenizer, max_length=1024):
#     """Generate a resume given an instruction."""
#     inputs = tokenizer(
#         f"generate_resume: {instruction}",
#         return_tensors="pt",
#         max_length=max_length,
#         truncation=True,
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=512,
#         top_k=50,                 #  top-k sampling
#         top_p=0.95, 
#         temperature=5,            # Controlled randomness
#         repetition_penalty=3.0,     # Higher penalty for repetition
#         num_beams=4,                # beam search for diversity
#         no_repeat_ngram_size=3,     # Penalizing repeated n-grams
#         early_stopping=True         # Stopping when EOS token is reached
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def format_resume(text, sections):
#     """Format the resume text based on section headers."""
#     for section in sections:
#         text = re.sub(rf"(?i)({section}):", rf"\n{section.upper()}:", text)
#     text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
#     return text.strip()

# #%% 

# ## bleu score

# # import sacrebleu
# # from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# # from datasets import load_metric
# # model_path = "../models_bart2" 
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
# # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # def calculate_bleu(reference, hypothesis):
# #     """Calculate BLEU score between reference and hypothesis."""
# #     reference = [reference]
# #     hypothesis = [hypothesis]
# #     bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
# #     return bleu.score
# # bleu_scores = []
# # for example in tokenized_test.shuffle(seed=42).select(range(10)):
# #     reference_resume = example["Resume_test"]
# #     instruction = example["instruction"]
# #     generated_resume = generate_resume(instruction, model, tokenizer)
# #     bleu_score = calculate_bleu(reference_resume, generated_resume)
# #     bleu_scores.append(bleu_score)

# # average_bleu_score = sum(bleu_scores) / len(bleu_scores)
# # print(f"Average BLEU score on the test dataset: {average_bleu_score}")

# # rouge score
# # 
# # rouge = load_metric("rouge")

# # def calculate_rouge(reference, hypothesis):
# #     """Calculate ROUGE score between reference and hypothesis."""
# #     scores = rouge.compute(predictions=[hypothesis], references=[reference])
# #     return scores

# # rouge_scores = []
# # for example in tokenized_test.shuffle(seed=42).select(range(10)):
# #     reference_resume = example["Resume_test"]
# #     instruction = example["instruction"]
# #     generated_resume = generate_resume(instruction, model, tokenizer)
# #     rouge_score = calculate_rouge(reference_resume, generated_resume)
# #     rouge_scores.append(rouge_score)

# # # Calculating average ROUGE scores
# # average_rouge_score = {
# #     "rouge1": sum(score["rouge1"].mid.fmeasure for score in rouge_scores) / len(rouge_scores),
# #     "rouge2": sum(score["rouge2"].mid.fmeasure for score in rouge_scores) / len(rouge_scores),
# #     "rougeL": sum(score["rougeL"].mid.fmeasure for score in rouge_scores) / len(rouge_scores),
# # }

# # print(f"Average ROUGE-1 score on the test dataset: {average_rouge_score['rouge1']}")
# # print(f"Average ROUGE-2 score on the test dataset: {average_rouge_score['rouge2']}")
# # print(f"Average ROUGE-L score on the test dataset: {average_rouge_score['rougeL']}")


# #%% Step 11: Generate and clean a resume 
# instruction = "Generate a Resume for a Systems Administrator Job"
# generated_resume = generate_resume(instruction, model, tokenizer)
# # Display the cleaned resume
# print("Cleaned Resume:")
# print(generated_resume)


# # %%
# ################################################################################################################################
# # This code is for testing the resume generation
# ################################################################################################################################
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from datasets import load_dataset
# from collections import Counter
# from datasets import DatasetDict

# # device
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the model and tokenizer
# model_path = "../models_bart"  # Replace with model path
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# def generate_resume(instruction, model, tokenizer, max_length=1024):
#     """Generate a resume given an instruction."""
#     inputs = tokenizer(
#         f"generate_resume: {instruction}",
#         return_tensors="pt",
#         max_length=max_length,
#         truncation=True,
#     )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     inputs = {key: value.to(device) for key, value in inputs.items()}

    
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=1024,
#         do_sample=True,
#         top_k=50,                
#         top_p=0.9, 
#         temperature=.7,            
#         repetition_penalty=5.0,     
#         num_beams=6,                
#         no_repeat_ngram_size=5,     
#         early_stopping=True         
#     )
# #     outputs = model.generate(
# #     inputs["input_ids"],
# #     max_length=512,
# #     num_beams=5,
# #     no_repeat_ngram_size=3,
# #     early_stopping=True
# # )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# #not used in the latest version
# def clean_resume(text):
#     """Apply additional cleaning and formatting for better readability."""
#     # Removing duplicate lines
#     lines = text.split("\n")
#     seen = set()
#     cleaned_lines = []
#     for line in lines:
#         if line not in seen:
#             seen.add(line)
#             cleaned_lines.append(line)

#     text = "\n".join(cleaned_lines)

#     # Removing excessive whitespace
#     text = re.sub(r"\s+", " ", text)
#     sections = ["Professional Summary", "Work Experience", "Education", "Skills"]
#     for section in sections:
#         text = re.sub(section, f"\n{section.upper()}\n", text, flags=re.IGNORECASE)
#     text = re.sub(r"(?<!\n)-\s*", "\n- ", text)

#     return text.strip()
# # %%
# def format_resume(text, sections):
#     """Format the resume text based on section headers."""
#     # Capitalize and format sections for better readability
#     # text = re.sub(r"(?i)([A-Z][A-Z\s]*):", r"\n\1\n", text)
#     for section in sections:
#         text = re.sub(rf"(?i)\b({section})\b:", rf"\n{section.upper()}:", text)
#     text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
    
#     return text.strip()

# # %%
# instruction = "Generate a Resume for a Accountant Job"
# generated_resume = generate_resume(instruction, model, tokenizer)
# section_headers = [
#     "Professional Summary", "Summary","SKILL SET", "Objective", "Experience","EXPERIENCE", "Work History",
#     "Education", "Hobbies","SOFTWARE SKILLS","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
#     "Affiliations","TECHNICALSKILLS","TECHNICAL PROFICIENCIES","Additional Information SKILLS","SUMMARY OF SKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
# ]
# cleaned_resume = format_resume(generated_resume,section_headers)
# # Display the cleaned resume
# print("Cleaned Resume:")
# print(cleaned_resume)
# # %%

# #################################################################################################################################
# # Unused code
# #################################################################################################################################


# #counts the instruction and avgt length in dataset for each type of prompt
# # 
# # instruction_counts = Counter(dataset['train']['instruction'])
# # print("Instruction counts:")

# # for instruction, count in instruction_counts.items():
# #     print(f"{instruction}: {count}")
    
# #     
# #     def average_resume_length(dataset):
# #         total_length = 0
# #         num_resumes = 0

# #         for resume in dataset['train']['Resume_test']:
# #             if resume:
# #                 total_length += len(resume.split())
# #                 num_resumes += 1

# #         return total_length / num_resumes if num_resumes > 0 else 0

# #     avg_length = average_resume_length(dataset)
# #     print(f"Average Resume_test length: {avg_length} words")



#########################################
# resume_train_preprocess_test.py
#########################################

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

# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from datasets import load_dataset

# class gen_resume: 
    
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # # model_path = "models_bart1"  
#     # model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
#     # tokenizer = AutoTokenizer.from_pretrained(model_path)

#     def generate_resume(self, instruction, model_path, max_length=1024):
#         """Generate a resume given an instruction."""
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)

#         inputs = tokenizer(
#             f"generate_resume: {instruction}",
#             return_tensors="pt",
#             max_length=max_length,
#             truncation=True,
#         )
#         # inputs = tokenizer(
#         #     f"generate_resume: {instruction}",
#         #     return_tensors="pt",
#         #     max_length=max_length,
#         #     truncation=True,
#         # )
#         # model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
#         # tokenizer = AutoTokenizer.from_pretrained(model_path)
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         outputs = model.generate(
#             inputs["input_ids"],
#             max_length=1024,
#             do_sample=True,
#             top_k=50,                 
#             top_p=0.9, 
#             temperature=.7,            
#             repetition_penalty=5.0,     
#             num_beams=6,                
#             no_repeat_ngram_size=5,     
#             early_stopping=True         
#         )
#         return tokenizer.decode(outputs[0], skip_special_tokens=True)

#     def clean_resume(self, text):
#         lines = text.split("\n")
#         seen = set()
#         cleaned_lines = []
#         for line in lines:
#             if line not in seen:
#                 seen.add(line)
#                 cleaned_lines.append(line)

#         text = "\n".join(cleaned_lines)
#         text = re.sub(r"\s+", " ", text)
#         sections = ["Professional Summary", "Work Experience", "Education", "Skills"]
#         for section in sections:
#             text = re.sub(section, f"\n{section.upper()}\n", text, flags=re.IGNORECASE)
#         text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
#         return text.strip()
    
#     # def format_resume(self, text, sections):
#     #     """Format the resume text based on section headers."""
#     #     for section in sections:
#     #         text = re.sub(rf"(?i)({section})\s*:", rf"\n{section.upper()}:", text)
#     #     text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
#     #     return text.strip()
    
#     def format_resume(text, sections):
#         """Format the resume text based on section headers."""
#         # Capitalize and format sections for better readability
#         # text = re.sub(r"(?i)([A-Z][A-Z\s]*):", r"\n\1\n", text)
#         for section in sections:
#             text = re.sub(rf"(?i)\b({section})\b:", rf"\n{section.upper()}:", text)
#         # Add bullet points for lists
#         text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
        
#         return text.strip()
    
#     def main(self, instruction):
#         section_headers = [
#     "Professional Summary", "Summary","SKILL SET", "Objective", "Experience", "Work History",
#     "Education", "Hobbies","SOFTWARE SKILLS","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
#     "Affiliations","TECHNICALSKILLS","TECHNICAL PROFICIENCIES","Additional Information SKILLS","SUMMARY OF SKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
# ]
#     # instruction = "Generate a resume for a software engineer job."
#         generated_resume = self.generate_resume(instruction, self.model, self.tokenizer)
#         cleaned_resume = self.format_resume(generated_resume, section_headers)
#         # print("Cleaned Resume:")
#         # print(cleaned_resume)
#         return cleaned_resume


#########################################
# skill_gap.py
#########################################
#%%

################################################################################################################
#  Below is the skill gap analysis that is being used in the the app
###############################################################################################################

# import json
# import spacy

# # spaCy model
# import spacy.cli
# spacy.cli.download("en_core_web_md")
# nlp = spacy.load("en_core_web_md")

# def analyze_skill_gap(resume_json, job_description_json):
    
#     # Parseing the JSON data
#     resume_data = resume_json
#     job_description_data =job_description_json

#     resume_skills = resume_data.get("skills", [])
#     job_skills = job_description_data.get("required_skills", [])

#     missing_skills = [skill for skill in job_skills if skill not in resume_skills]

#     resume_experience = [exp["responsibilities"] for exp in resume_data.get("work_experience", [])]
#     resume_experience = [item for sublist in resume_experience for item in sublist]  
#     resume_education = resume_data.get("education", [])
#     job_experience = job_description_data.get("responsibilities", [])
#     job_education = job_description_data.get("qualifications", [])

#     experience_gap = []
#     for job_exp in job_experience:
#         job_exp_doc = nlp(job_exp)
#         if not any(job_exp_doc.similarity(nlp(exp)) > 0.8 for exp in resume_experience):
#             experience_gap.append(job_exp)

#     education_gap = []
#     for job_edu in job_education:
#         job_edu_doc = nlp(job_edu)
#         if not any(job_edu_doc.similarity(nlp(edu["degree"])) > 0.8 for edu in resume_education):
#             education_gap.append(job_edu)

#     feedback = ""
#     if missing_skills:
#         feedback += f"The following skills are missing from your resume: {', '.join(missing_skills)}\n"
#     else:
#         feedback += "\nYour resume matches all the required skills for the job.\n"
    
#     if experience_gap:
#         feedback += f"\nThe following work experience is missing from your resume: {', '.join(experience_gap)}\n"
#     else:
#         feedback += "\nYour resume matches all the required work experiences for the job.\n"

#     if education_gap:
#         feedback += f"\nThe following educational qualifications are missing from your resume: {', '.join(education_gap)}\n"
#     else:
#         feedback += "\nYour resume matches all the required educational qualifications for the job.\n"

#     return feedback
# # resume_json = '''
# # {
# #     "skills": [
# #         "Python",
# #         "MachineLearning",
# #         "UtilizingChatGPTandotherAItools",
# #         "Blender3DTool",
# #         "Modelling",
# #         "Composting",
# #         "3Ddesigningandmodelling",
# #         "3DAnimation"
# #     ],
# #     "work_experience": [
# #         "2 years testing experience"
# #     ],
# #     "projects": [
# #         {
# #             "name": "AUTOMATICPERSONALITYRECOGNITIONINVIDEOINTERVIEWSUSINGCNN",
# #             "description": "Developed an end-to-end interviewing model to perform automatic personality recognition (APR) during interviews.\\nThrought the input of interview video, this model will do screening process based on 5 personality traits.(OCEAN model)\\nImplemented by using visual and audio subnetworks in this project.\\nThe dataset used is First Impression V2, it consist of 10000 video files along with annotation files.",
# #             "technologies": []
# #         },
# #         {
# #             "name": "INTELLIGENTWASTESEGREGATIONTECHNIQUEUSINGCNN",
# #             "description": "Developed a waste segregation model that can classify the waste into 9 different classes.\\nUsed a Deep Learning algorithm (VGG-16).\\nUsed MSW dataset and added our own images from google images to the dataset.",
# #             "technologies": []
# #         },
# #         {
# #             "name": "VISION",
# #             "description": "Developed an application with smart assistance for blind people.\\nUsed COCO dataset and ssdmobilenetv2model for detecting objects.\\nThis application will detect objects around the victim and gives audio output of the object detected.\\nDeveloped the application using android studio. ashik.shaik.ali@gmail.com +1(571)-413-4739",
# #             "technologies": []
# #         },
# #         {
# #             "name": "AUDIOTRANSMISSIONTHROUGHLASER",
# #             "description": "Developed a device that transmits audio through LASER.\\nThis works under the principle of intensity modulation and demodulation.\\nSolar plate is used as demodulator.",
# #             "technologies": []
# #         }
# #     ],
# #     "education": [
# #         {
# #             "institution": "GeorgeMasonUniversity",
# #             "degree": "MS",
# #             "graduation_year": "2024-25"
# #         },
# #         {
# #             "institution": "VardhamanCollegeofEngineering",
# #             "degree": "B-Tech",
# #             "graduation_year": "2019-23"
# #         },
# #         {
# #             "institution": "SriNalandaJuniorCollege",
# #             "degree": "XII",
# #             "graduation_year": "2017-19"
# #         },
# #         {
# #             "institution": "ReginaCarmeliConventHighSchool",
# #             "degree": "X",
# #             "graduation_year": "2005-17"
# #         }
# #     ]
# # }
# # '''

# # job_description_json = '''
# # {
# #     "skills": [
# #         "Python",
# #         "MachineLearning",
# #         "DeepLearning",
# #         "DataAnalysis",
# #         "TensorFlow",
# #         "Keras",
# #         "NLP"
# #     ],
# #     "work_experience": [
# #         "2+ years of experience in Machine Learning",
# #         "Experience with TensorFlow and Keras"
# #     ],
# #     "education": [
# #         {
# #             "institution": "Any accredited university",
# #             "degree": "MS",
# #             "graduation_year": "Any"
# #         }
# #     ]
# # }
# # '''

# # feedback = analyze_skill_gap(resume_json, job_description_json)
# # print(feedback)
# # %%

#######################################################
# Streamlit first version, now it is testapp2.py
#######################################################

# import streamlit as st
# import pandas as pd
# st.title("Resume and Job Matcher")
# st.sidebar.title("Menu")
# menu_options = ["Home", "About"]
# choice = st.sidebar.selectbox("Select an option", menu_options)
# if choice == "Home":
#     st.header("Job Description")
#     job_description = st.text_area("Paste the job description here")
#     st.header("Upload Resume")
#     uploaded_file = st.file_uploader("Drag and drop your resume here", type=["pdf", "docx"])
#     if uploaded_file is not None:
#         st.write("Uploaded Resume:", uploaded_file.name)
#         st.header("Output Results")
#         st.write("Results will be displayed here after processing.")
# elif choice == "About":
#     st.header("About")
#     st.write("This is a basic Streamlit application for matching resumes with job descriptions.")
