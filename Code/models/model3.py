#%% Import Libraries
import nltk
from nltk.corpus import stopwords
import re
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

#%% Initializing model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

#%% Preprocessing

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
        highlighted_text += f"{header.upper()}:\n"
        highlighted_text += content + "\n\n"
    return highlighted_text.strip()

def preprocess_resume(text, headers):
    sections = extract_sections(text, headers)
    highlighted_text = highlight_sections(sections)
    highlighted_text = re.sub(r'\s+', ' ', highlighted_text).strip()
    return highlighted_text

section_headers = [
    "Professional Summary", "Summary","SKILL SET", "Objective", "Experience", "Work History",
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
train_test_split = dataset.train_test_split(test_size=0.1)
tokenized_train = train_test_split["train"].map(preprocess_data, batched=True)
tokenized_test = train_test_split["test"].map(preprocess_data, batched=True)

#%%

# test oversampling
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from datasets import Dataset

def make_unique(column_names):
    seen = set()
    result = []
    for col in column_names:
        new_col = col
        count = 1
        while new_col in seen:
            new_col = f"{col}_{count}"
            count += 1
        seen.add(new_col)
        result.append(new_col)
    return result

# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset["train"])

# Check for duplicate columns
print("Columns before renaming:")
print(df.columns)

# Rename duplicate columns
df.columns = make_unique(df.columns)

# Check for duplicate columns after renaming
print("Columns after renaming:")
print(df.columns)

# Count the occurrences of each instruction
instruction_counts = Counter(df["instruction"])

# Identify the minority instructions
print("Instruction counts before oversampling:")
print(instruction_counts)

# Prepare the data for oversampling
X = df.drop(columns=["Resume_test"])
y = df["instruction"]

# Apply RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Convert the resampled data back to a DataFrame
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Ensure there are no duplicate columns after resampling
df_resampled.columns = make_unique(df_resampled.columns)

# Convert the DataFrame back to the original dataset format
oversampled_dataset = Dataset.from_pandas(df_resampled)

# Verify the new instruction counts
new_instruction_counts = Counter(df_resampled["instruction"])
print("Instruction counts after oversampling:")
print(new_instruction_counts)
#%%
dataset = oversampled_dataset


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
    num_train_epochs=1,
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
trainer.save_model('/home/ubuntu/Project/NLP_FinalProject-Group1/Code/models_bart1')


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
#%%
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
model_path = "/home/ubuntu/Project/NLP_FinalProject-Group1/Code/models_bart" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score between reference and hypothesis."""
    reference = [reference]
    hypothesis = [hypothesis]
    bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
    return bleu.score
bleu_scores = []
for example in tokenized_test.shuffle(seed=42).select(range(500)):
    reference_resume = example["Resume_test"]
    instruction = example["instruction"]
    generated_resume = generate_resume(instruction, model, tokenizer)
    bleu_score = calculate_bleu(reference_resume, generated_resume)
    bleu_scores.append(bleu_score)

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score on the test dataset: {average_bleu_score}")


#%% Step 11: Generate and clean a resume ctesst
instruction = "Generate a professional resume for a Accountant job."
generated_resume = generate_resume(instruction, model, tokenizer)
# Display the cleaned resume
print("Cleaned Resume:")
print(generated_resume)


# # %%
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from datasets import load_dataset
# from collections import Counter

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the model and tokenizer
# model_path = "/home/ubuntu/Project/NLP_FinalProject-Group1/Code/models_bart"  # Replace with your saved model path
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# def generate_resume(instruction, model, tokenizer, max_length=1024):
#     """Generate a resume given an instruction."""
#     # Tokenize the input instruction
#     inputs = tokenizer(
#         f"generate_resume: {instruction}",
#         return_tensors="pt",
#         max_length=max_length,
#         truncation=True,
#     )

#     # Move inputs to the same device as the model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     # Generate text with controlled randomness and repetition penalties
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=1024,
#         do_sample=True,
#         top_k=50,                 # Use top-k sampling
#         top_p=0.9, 
#         temperature=.7,            # Add controlled randomness
#         repetition_penalty=5.0,     # Higher penalty for repetition
#         num_beams=6,                # Use beam search for diversity
#         no_repeat_ngram_size=7,     # Penalize repeated n-grams
#         early_stopping=True         # Stop when EOS token is reached
#     )
# #     outputs = model.generate(
# #     inputs["input_ids"],
# #     max_length=512,
# #     num_beams=5,
# #     no_repeat_ngram_size=3,
# #     early_stopping=True
# # )

#     # Decode and return output
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def clean_resume(text):
#     """Apply additional cleaning and formatting for better readability."""
#     # Remove duplicate lines
#     lines = text.split("\n")
#     seen = set()
#     cleaned_lines = []
#     for line in lines:
#         if line not in seen:
#             seen.add(line)
#             cleaned_lines.append(line)

#     text = "\n".join(cleaned_lines)

#     # Remove excessive whitespace
#     text = re.sub(r"\s+", " ", text)

#     # Capitalize sections for better readability
#     sections = ["Professional Summary", "Work Experience", "Education", "Skills"]
#     for section in sections:
#         text = re.sub(section, f"\n{section.upper()}\n", text, flags=re.IGNORECASE)

#     # Add bullet points for lists
#     text = re.sub(r"(?<!\n)-\s*", "\n- ", text)

#     return text.strip()
# # %%
# def format_resume(text, sections):
#     """Format the resume text based on section headers."""
#     # Capitalize and format sections for better readability
#     # text = re.sub(r"(?i)([A-Z][A-Z\s]*):", r"\n\1\n", text)
#     for section in sections:
#         text = re.sub(rf"(?i)({section}):", rf"\n{section.upper()}:", text)
#     # Add bullet points for lists
#     text = re.sub(r"(?<!\n)-\s*", "\n- ", text)
    
#     return text.strip()

# # %%
# instruction = "Generate a Resume for a Sales Job"
# generated_resume = generate_resume(instruction, model, tokenizer)
# #%%
# cleaned_resume = format_resume(generated_resume,section_headers)


# # Display the cleaned resume
# print("Cleaned Resume:")
# print(cleaned_resume)
# # %%
# # Check the count of instructions in the dataset
# instruction_counts = Counter(dataset['train']['instruction'])

# # Display the count of each instruction type
# print("Instruction counts:")
# for instruction, count in instruction_counts.items():
#     print(f"{instruction}: {count}")
    
#     # Calculate the average length of 'Resume_test' in the dataset
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
