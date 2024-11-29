
#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.nn.utils.rnn import pad_sequence
#%%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
base_dir= os.path.abspath('../..')
resume_dir = os.path.join(base_dir,'assests','data','resume_data.json')
job_dir = os.path.join(base_dir,'assests','data','jobs_data.json')

with open(job_dir, "r") as f:
    job_descriptions = json.load(f)


with open(resume_dir, "r") as f:
    resumes = json.load(f)
#%%
# Extracting relevant features
job_texts = [
    " ".join([
        " ".join(job["skills_required"]),
        str(job["experience_required"]["min_years"]),
        str(job["experience_required"]["max_years"]),
        job["qualifications"],
        job["responsibilities"]
    ]) for job in job_descriptions
]

resume_texts = [
    " ".join([
        " ".join(resume["skills"]["technical_skills"]),
        " ".join(resume["experience"]["details"]),
        " ".join(resume["education"]["details"]),
        resume["responsibilities"],
        resume["summary"]
    ]) for resume in resumes
]
#%%

job_titles = [job["title"] for job in job_descriptions]
label_encoder = LabelEncoder()
job_labels = label_encoder.fit_transform(job_titles)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(job_texts + resume_texts)

job_sequences = tokenizer.texts_to_sequences(job_texts)
resume_sequences = tokenizer.texts_to_sequences(resume_texts)

max_length = max(max(len(seq) for seq in job_sequences), max(len(seq) for seq in resume_sequences))
job_padded = pad_sequence([torch.tensor(seq) for seq in job_sequences], batch_first=True, padding_value=0)
resume_padded = pad_sequence([torch.tensor(seq) for seq in resume_sequences], batch_first=True, padding_value=0)
#%%
class JobDescriptionDataset(Dataset):
    def __init__(self, text_data, labels):
        self.text_data = text_data
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_data[idx], self.labels[idx]

job_train, job_test, label_train, label_test = train_test_split(job_padded, job_labels, test_size=0.2, random_state=42)

train_dataset = JobDescriptionDataset(job_train, label_train)
test_dataset = JobDescriptionDataset(job_test, label_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#%%
# Model
class JobDescriptionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(JobDescriptionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_text, return_hidden=False):
        embedded = self.embedding(input_text)
        _, (hidden, _) = self.lstm(embedded)
        if return_hidden:
            return hidden.squeeze(0)  
        output = self.fc(hidden.squeeze(0)) 
        return output
#%%
# Model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
hidden_dim = 128
num_classes = len(label_encoder.classes_)

model = JobDescriptionModel(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%%
# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_text, labels in train_loader:
        input_text, labels = input_text.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "job_description_model.pth")
print("Model saved!")
#%%
# Evaluation
from sklearn.metrics import cohen_kappa_score
model.eval()
correct = 0
total = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for input_text, labels in test_loader:
        input_text, labels = input_text.to(device), labels.to(device)
        outputs = model(input_text)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
kappa_score = cohen_kappa_score(all_labels, all_predictions)
print(f"Test Accuracy: {correct / total:.4f}")
print(f"Cohen's Kappa: {kappa_score:.4f}")
#%%
def get_embedding(model, text_sequence):
    model.eval()
    with torch.no_grad():
        input_tensor = pad_sequence([torch.tensor(text_sequence)], batch_first=True, padding_value=0).to(device)
        embedding = model(input_tensor, return_hidden=True)  
    return embedding.cpu().numpy()

from sklearn.metrics.pairwise import cosine_similarity

resume_text_sequence = tokenizer.texts_to_sequences([" ".join(resume_texts[0])])[0]
job_text_sequence = tokenizer.texts_to_sequences([" ".join(job_texts[0])])[0]

resume_embedding = get_embedding(model, resume_text_sequence)
job_embedding = get_embedding(model, job_text_sequence)

similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
print(f"Similarity Score: {similarity_score:.4f}")


def provide_feedback(resume, job_description, tokenizer, model):

    resume_sequence = tokenizer.texts_to_sequences([" ".join(resume)])[0]
    job_sequence = tokenizer.texts_to_sequences([" ".join(job_description)])[0]
    
    resume_embedding = get_embedding(model, resume_sequence)
    job_embedding = get_embedding(model, job_sequence)
    

    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    
    missing_skills = set(job_description["skills_required"]) - set(resume["skills"]["technical_skills"])
    
    feedback = {
        "similarity_score": similarity_score,
        "missing_skills": list(missing_skills),
        "recommendation": f"Consider adding the following skills: {', '.join(missing_skills)}"
    }
    return feedback

feedback = provide_feedback(resumes[0], job_descriptions[0], tokenizer, model)
print("Feedback:", feedback)

#%%




#%%

from sklearn.metrics.pairwise import cosine_similarity

def encode_text(model, text_input):
    model.eval()
    with torch.no_grad():
        text_seq = tokenizer.texts_to_sequences([text_input])
        text_tensor = pad_sequence([torch.tensor(text_seq[0])], batch_first=True, padding_value=0).to(device)
        embedded = model.embedding(text_tensor)
        _, (hidden, _) = model.lstm(embedded)
        return hidden.squeeze(0).cpu().numpy()

def compare_resume_to_job(resume_text, job_text):
    resume_embedding = encode_text(model, resume_text)
    job_embedding = encode_text(model, job_text)
    resume_embedding_flat = resume_embedding.flatten()
    job_embedding_flat = job_embedding.flatten()
    similarity = cosine_similarity([resume_embedding_flat], [job_embedding_flat])[0][0]
    return similarity
#%%
# Feedback function
def provide_feedback(resume, job_description):
    resume_text = " ".join([
        " ".join(resume["skills"]["technical_skills"]),
        " ".join(resume["experience"]["details"]),
        " ".join(resume["education"]["details"]),
        resume["responsibilities"],
        resume["summary"]
    ])
    job_text = " ".join([
        " ".join(job_description["skills_required"]),
        str(job_description["experience_required"]["min_years"]),
        str(job_description["experience_required"]["max_years"]),
        job_description["qualifications"],
        job_description["responsibilities"]
    ])
    
    similarity = compare_resume_to_job(resume_text, job_text)
    missing_skills = set(job_description["skills_required"]) - set(resume["skills"]["technical_skills"])

    feedback = {
        "similarity_score": similarity,
        "missing_skills": list(missing_skills)
    }
    return feedback


resume = resumes[0]
job = job_descriptions[0]
feedback = provide_feedback(resume, job)
print("Feedback:", feedback)

# %%
