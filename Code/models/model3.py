#%% Import Libraries
import nltk
from nltk.corpus import stopwords
import re
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
from datasets import DatasetDict, Dataset

#%% Step 1: Load Dataset
dataset = load_dataset("InferencePrince555/Resume-Dataset")

# Cleaning
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"<.*?>", "", text)  # HTML tags
    text = re.sub(r"span l.*?span", "", text)  # invalid spans
    text = re.sub(r"\s+", " ", text.strip())  # extra whitespace
    return text

dataset = dataset.map(lambda x: {"Resume_test": clean_text(x["Resume_test"])})

#%%

# def resample_dataset(dataset, target_count):
#     instruction_counts = Counter(dataset['train']['instruction'])
#     instruction_datasets = {instruction: [] for instruction in instruction_counts}
#     for example in dataset['train']:
#         instruction_datasets[example['instruction']].append(example)
#     resampled_data = []
#     for examples in instruction_datasets.values():
#         if len(examples) < target_count:
#             resampled_data.extend(examples * (target_count // len(examples)) + examples[:target_count % len(examples)])
#         else:
#             resampled_data.extend(examples)
#     resampled_dataset = Dataset.from_dict({key: [example[key] for example in resampled_data] for key in resampled_data[0]})
    
#     return DatasetDict({'train': resampled_dataset})
# target_count = 1500
# dataset = resample_dataset(dataset, target_count)

#%% Initializing model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

#%% Preprocessing

def extract_sections(text, headers):
    sections = {}
    header_pattern = r'\b(' + '|'.join(re.escape(header) for header in headers) + r')\b'
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
        highlighted_text += f"{header.upper()}:\n"
        highlighted_text += content + "\n\n"
    return highlighted_text.strip()

def preprocess_resume(text, headers):
    sections = extract_sections(text, headers)
    highlighted_text = highlight_sections(sections)
    highlighted_text = re.sub(r'\s+', ' ', highlighted_text).strip()
    return highlighted_text

section_headers = [
    "Professional Summary", "Summary","SKILL SET", "Objective", "Experience","EXPERIENCE", "Work History",
    "Education", "Hobbies","SOFTWARE SKILLS","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
    "Affiliations","TECHNICALSKILLS","TECHNICAL PROFICIENCIES","Additional Information SKILLS","SUMMARY OF SKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
]

#%%

def preprocess_dataset(dataset, headers):
    def preprocess_row(row):
        if row['Resume_test'] is None:
            return {"Resume_test": ""}
        return {"Resume_test": preprocess_resume(row['Resume_test'], headers)}
    return dataset.map(preprocess_row)
dataset = preprocess_dataset(dataset, section_headers)

# first and last row of dataset

# print("First row of dataset:")
# print(dataset['train'][0])

# print("Last row of dataset:")
# print(dataset['train'][-1])
#%%
def preprocess_data(examples):

    inputs = tokenizer(examples["instruction"], max_length=1024, truncation=True, padding="max_length")
    outputs = tokenizer(examples["Resume_test"], max_length=1024, truncation=True, padding="max_length")
    
    inputs["labels"] = outputs["input_ids"]
    return inputs

# Apply preprocessing and data split
# tokenized_dataset = dataset["train"].map(preprocess_data, batched=True)
train_test_split = dataset['train'].train_test_split(test_size=0.1)
tokenized_train = train_test_split["train"].map(preprocess_data, batched=True)
tokenized_test = train_test_split["test"].map(preprocess_data, batched=True)


#%% Load Model
model = BartForConditionalGeneration.from_pretrained(model_name)
# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#%% Training Arguments
# training_args = TrainingArguments(
#     output_dir="./resume_generator_bart",
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     save_strategy="epoch",
#     logging_dir="./logs",
#     logging_steps=50,
#     save_total_limit=2
# )


training_args = TrainingArguments(
    output_dir="./resume_generator_bart",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    fp16=True,  
    gradient_accumulation_steps=4  
)



# Initializing Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator
)

#%% Training the Model
trainer.train()

#%% Saving the Model
trainer.save_model('../models_bart2')

#%% Generate and and eval
def generate_resume(instruction, model, tokenizer, max_length=1024):
    """Generate a resume given an instruction."""
    inputs = tokenizer(
        f"generate_resume: {instruction}",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        top_k=50,                 #  top-k sampling
        top_p=0.95, 
        temperature=5,            # Controlled randomness
        repetition_penalty=3.0,     # Higher penalty for repetition
        num_beams=4,                # beam search for diversity
        no_repeat_ngram_size=3,     # Penalizing repeated n-grams
        early_stopping=True         # Stopping when EOS token is reached
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def format_resume(text, sections):
    """Format the resume text based on section headers."""
    for section in sections:
        text = re.sub(rf"(?i)({section}):", rf"\n{section.upper()}:", text)
    text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
    return text.strip()

#%% 

## bleu score

# import sacrebleu
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from datasets import load_metric
# model_path = "../models_bart2" 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# def calculate_bleu(reference, hypothesis):
#     """Calculate BLEU score between reference and hypothesis."""
#     reference = [reference]
#     hypothesis = [hypothesis]
#     bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
#     return bleu.score
# bleu_scores = []
# for example in tokenized_test.shuffle(seed=42).select(range(10)):
#     reference_resume = example["Resume_test"]
#     instruction = example["instruction"]
#     generated_resume = generate_resume(instruction, model, tokenizer)
#     bleu_score = calculate_bleu(reference_resume, generated_resume)
#     bleu_scores.append(bleu_score)

# average_bleu_score = sum(bleu_scores) / len(bleu_scores)
# print(f"Average BLEU score on the test dataset: {average_bleu_score}")

# rouge score
# 
# rouge = load_metric("rouge")

# def calculate_rouge(reference, hypothesis):
#     """Calculate ROUGE score between reference and hypothesis."""
#     scores = rouge.compute(predictions=[hypothesis], references=[reference])
#     return scores

# rouge_scores = []
# for example in tokenized_test.shuffle(seed=42).select(range(10)):
#     reference_resume = example["Resume_test"]
#     instruction = example["instruction"]
#     generated_resume = generate_resume(instruction, model, tokenizer)
#     rouge_score = calculate_rouge(reference_resume, generated_resume)
#     rouge_scores.append(rouge_score)

# # Calculating average ROUGE scores
# average_rouge_score = {
#     "rouge1": sum(score["rouge1"].mid.fmeasure for score in rouge_scores) / len(rouge_scores),
#     "rouge2": sum(score["rouge2"].mid.fmeasure for score in rouge_scores) / len(rouge_scores),
#     "rougeL": sum(score["rougeL"].mid.fmeasure for score in rouge_scores) / len(rouge_scores),
# }

# print(f"Average ROUGE-1 score on the test dataset: {average_rouge_score['rouge1']}")
# print(f"Average ROUGE-2 score on the test dataset: {average_rouge_score['rouge2']}")
# print(f"Average ROUGE-L score on the test dataset: {average_rouge_score['rougeL']}")


#%% Step 11: Generate and clean a resume 
instruction = "Generate a Resume for a Systems Administrator Job"
generated_resume = generate_resume(instruction, model, tokenizer)
# Display the cleaned resume
print("Cleaned Resume:")
print(generated_resume)


# %%
################################################################################################################################
# This code is for testing the resume generation
################################################################################################################################
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from collections import Counter
from datasets import DatasetDict

# device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_path = "../models_bart"  # Replace with model path
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_resume(instruction, model, tokenizer, max_length=1024):
    """Generate a resume given an instruction."""
    inputs = tokenizer(
        f"generate_resume: {instruction}",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
#     outputs = model.generate(
#     inputs["input_ids"],
#     max_length=512,
#     num_beams=5,
#     no_repeat_ngram_size=3,
#     early_stopping=True
# )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#not used in the latest version
def clean_resume(text):
    """Apply additional cleaning and formatting for better readability."""
    # Removing duplicate lines
    lines = text.split("\n")
    seen = set()
    cleaned_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Removing excessive whitespace
    text = re.sub(r"\s+", " ", text)
    sections = ["Professional Summary", "Work Experience", "Education", "Skills"]
    for section in sections:
        text = re.sub(section, f"\n{section.upper()}\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\n)-\s*", "\n- ", text)

    return text.strip()
# %%
def format_resume(text, sections):
    """Format the resume text based on section headers."""
    # Capitalize and format sections for better readability
    # text = re.sub(r"(?i)([A-Z][A-Z\s]*):", r"\n\1\n", text)
    for section in sections:
        text = re.sub(rf"(?i)\b({section})\b:", rf"\n{section.upper()}:", text)
    text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
    
    return text.strip()

# %%
instruction = "Generate a Resume for a Accountant Job"
generated_resume = generate_resume(instruction, model, tokenizer)
section_headers = [
    "Professional Summary", "Summary","SKILL SET", "Objective", "Experience","EXPERIENCE", "Work History",
    "Education", "Hobbies","SOFTWARE SKILLS","Education Details","Training attended","TECHNICAL EXPERTISE","Technical Expertise",  "SKILLS","CORE COMPETENCIES", "Skills", "Certifications", "Projects", "Accomplishments",
    "Affiliations","TECHNICALSKILLS","TECHNICAL PROFICIENCIES","Additional Information SKILLS","SUMMARY OF SKILLS","Technical Summary","Computer skills","Key Skills","TECHNICAL STRENGTHS","Technical Skill Set",  "KEY COMPETENCIES","PERSONAL SKILLS","IT SKILLS","Skill Set","Areas of expertise","AREA OF EXPERTISE", "Interests", "Languages", "References", "Technical Skills"
]
cleaned_resume = format_resume(generated_resume,section_headers)
# Display the cleaned resume
print("Cleaned Resume:")
print(cleaned_resume)
# %%

#################################################################################################################################
# Unused code
#################################################################################################################################


#counts the instruction and avgt length in dataset for each type of prompt
# 
# instruction_counts = Counter(dataset['train']['instruction'])
# print("Instruction counts:")

# for instruction, count in instruction_counts.items():
#     print(f"{instruction}: {count}")
    
#     
#     def average_resume_length(dataset):
#         total_length = 0
#         num_resumes = 0

#         for resume in dataset['train']['Resume_test']:
#             if resume:
#                 total_length += len(resume.split())
#                 num_resumes += 1

#         return total_length / num_resumes if num_resumes > 0 else 0

#     avg_length = average_resume_length(dataset)
#     print(f"Average Resume_test length: {avg_length} words")
# %%
