# #%%
# from datasets import load_dataset
# from transformers import AutoTokenizer, BartTokenizer
# from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
# from transformers import DataCollatorForSeq2Seq
# #%%
# # Load the dataset
# dataset = load_dataset("InferencePrince555/Resume-Dataset")

# 
# print(dataset)

# # %%
# model_name = "facebook/bart-large"  
# tokenizer = BartTokenizer.from_pretrained(model_name)

# def preprocess_data(examples):
#     
#     instructions = [instruction if instruction else "" for instruction in examples['instruction']]
#     resumes = [resume if resume else "" for resume in examples['Resume_test']]
    
#    
#     inputs = tokenizer(instructions, max_length=512, truncation=True, padding="max_length")
#     outputs = tokenizer(resumes, max_length=512, truncation=True, padding="max_length")
    
#     
#     inputs['labels'] = outputs['input_ids']
#     return inputs
# tokenized_dataset = dataset['train'].map(preprocess_data, batched=True)
# train_test_split = dataset['train'].train_test_split(test_size=0.1)
# tokenized_train = train_test_split['train'].map(preprocess_data, batched=True)
# tokenized_test = train_test_split['test'].map(preprocess_data, batched=True)

# #%%
# 
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 
# training_args = TrainingArguments(
#     output_dir="./resume_generator",
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     save_strategy="epoch",
#     logging_dir="./logs",
#     logging_steps=50,
#     save_total_limit=2
# )

# 
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_test,
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# 
# trainer.train()

# #%%
# trainer.save_model('/home/ubuntu/NLP/Project/NLP_FinalProject-Group1/Code/models')
# #%%
# 
# import torch

# def generate_resume(instruction, model, tokenizer, max_length=512):
#     
#     inputs = tokenizer(
#         f"generate_resume: {instruction}",
#         return_tensors="pt",
#         max_length=max_length,
#         truncation=True,
#     )
    
#     
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     inputs = {key: value.to(device) for key, value in inputs.items()}
    
#     
#     outputs = model.generate(
#         inputs['input_ids'],
#         max_length=512,
#         num_return_sequences=1,
#         temperature=0.7,            
#         repetition_penalty=2.0,    
#         top_k=50,                  
#         top_p=0.95                 #
#     )
    
#     t
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# instruction = "Generate a Resume for a Software Engineer Job"
# generated_resume = generate_resume(instruction, model, tokenizer)
# print("Generated Resume:")
# print(generated_resume)


# # %%


#%%
#%%
#%% Import 
import nltk
from nltk.corpus import stopwords
import re
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch

#%% Step 1: Load Dataset
dataset = load_dataset("InferencePrince555/Resume-Dataset")

# cleaning
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"<.*?>", "", text)  
    text = re.sub(r"span l.*?span", "", text)  
    text = re.sub(r"\s+", " ", text.strip())  
    return text

dataset = dataset.map(lambda x: {"Resume_test": clean_text(x["Resume_test"])})

#%% 
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

#%% Preprocess 


def preprocess_data(examples):

    inputs = tokenizer(examples["instruction"], max_length=512, truncation=True, padding="max_length")
    outputs = tokenizer(examples["Resume_test"], max_length=512, truncation=True, padding="max_length")
    
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset["train"].map(preprocess_data, batched=True)
train_test_split = dataset["train"].train_test_split(test_size=0.1)
tokenized_train = train_test_split["train"].map(preprocess_data, batched=True)
tokenized_test = train_test_split["test"].map(preprocess_data, batched=True)

#%% 
model = BartForConditionalGeneration.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#%% 
training_args = TrainingArguments(
    output_dir="./resume_generator_bart",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator
)

#%% 
trainer.train()

#%% 
trainer.save_model('/home/ubuntu/NLP/Project/NLP_FinalProject-Group1/Code/models_bart')

#%% 
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
        top_k=50,                 
        top_p=0.95, 
        temperature=5,            
        repetition_penalty=3.0,     
        num_beams=4,                
        no_repeat_ngram_size=3,     
        early_stopping=True         
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#%% 
def clean_resume(text):
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

#%% S
instruction = "Generate a professional resume for a accountant from India."
generated_resume = generate_resume(instruction, model, tokenizer)
cleaned_resume = clean_resume(generated_resume)
print("Cleaned Resume:")
print(cleaned_resume)


# %% Test
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "/home/ubuntu/NLP/Project/NLP_FinalProject-Group1/Code/models_bart"  
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
        max_length=700,
        do_sample=True,
        top_k=50,                 
        top_p=0.9, 
        temperature=.7,            
        repetition_penalty=5.0,     
        num_beams=6,                
        no_repeat_ngram_size=7,     
        early_stopping=True         
    )

    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_resume(text):
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

# %%
instruction = "Generate a resume for a software engineer job."
generated_resume = generate_resume(instruction, model, tokenizer)
cleaned_resume = clean_resume(generated_resume)
print("Cleaned Resume:")
print(cleaned_resume)
# %%
