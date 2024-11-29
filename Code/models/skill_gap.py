#%%
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score,  cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support


#%%
base_dir= os.path.abspath('../..')
resume_dir = os.path.join(base_dir,'assests','data','resume_data.json')
job_dir = os.path.join(base_dir,'assests','data','jobs_data.json')

with open(job_dir, "r") as f:
    job_descriptions = json.load(f)


with open(resume_dir, "r") as f:
    resumes = json.load(f)
#%%

job_skills = [" ".join(job["skills_required"]) for job in job_descriptions]
resume_skills = [" ".join(resume["skills"]["technical_skills"]) for resume in resumes]
#%%

tokenizer = Tokenizer()
tokenizer.fit_on_texts(job_skills + resume_skills)
job_sequences = tokenizer.texts_to_sequences(job_skills)
resume_sequences = tokenizer.texts_to_sequences(resume_skills)

max_length = max(max(len(seq) for seq in job_sequences), max(len(seq) for seq in resume_sequences))
job_padded = pad_sequence([torch.tensor(seq) for seq in job_sequences], batch_first=True, padding_value=0)
resume_padded = pad_sequence([torch.tensor(seq) for seq in resume_sequences], batch_first=True, padding_value=0)
#%%
from sklearn.preprocessing import LabelEncoder


job_titles = [job["title"] for job in job_descriptions]
resume_titles = [resume["job_title"] for resume in resumes]


all_titles = job_titles + resume_titles
label_encoder = LabelEncoder()
label_encoder.fit(all_titles)


job_labels = label_encoder.transform(job_titles)
resume_labels = label_encoder.transform(resume_titles)

#%%

labels = np.concatenate((job_labels, resume_labels))

class SkillGapDataset(Dataset):
    def __init__(self, job_data, resume_data, labels):
        self.job_data = torch.tensor(job_data, dtype=torch.long)
        self.resume_data = torch.tensor(resume_data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.job_data[idx], self.resume_data[idx], self.labels[idx]

#%%

class SkillGapModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_length):
        super(SkillGapModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        #
        self.fc = nn.Linear(2 * hidden_dim, len(label_encoder.classes_))  
        self.softmax = nn.Softmax(dim=1)

    def forward(self, job_input, resume_input):
        job_embedded = self.embedding(job_input)
        resume_embedded = self.embedding(resume_input)

        _, (job_hidden, _) = self.lstm(job_embedded)
        _, (resume_hidden, _) = self.lstm(resume_embedded)

        merged = torch.cat((job_hidden.squeeze(0), resume_hidden.squeeze(0)), dim=1)
        output = self.fc(merged)
        return self.softmax(output)  
#%%

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
hidden_dim = 128


model = SkillGapModel(vocab_size, embedding_dim, hidden_dim, max_length)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%%
# Training Loop

min_samples = min(len(job_padded), len(resume_padded), len(labels))

job_padded = job_padded[:min_samples]
resume_padded = resume_padded[:min_samples]
labels = labels[:min_samples]

job_train, job_test, resume_train, resume_test, label_train, label_test = train_test_split(
    job_padded, resume_padded, labels, test_size=0.2, random_state=42
)


train_dataset = SkillGapDataset(job_train, resume_train, label_train)
test_dataset = SkillGapDataset(job_test, resume_test, label_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
epochs =20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for job_input, resume_input, label in train_loader:
        optimizer.zero_grad()
        outputs = model(job_input, resume_input)
        loss = criterion(outputs.squeeze(), label.long())  
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    model_save_path = "skill_gap_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
#%%
from sklearn.metrics import classification_report
model.eval()
total_test_loss = 0
correct = 0



all_labels = []
all_predictions = []

with torch.no_grad():
    for job_input, resume_input, label in test_loader:
        outputs = model(job_input, resume_input)
        _, predicted = torch.max(outputs, dim=1) 

        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))

# Cohen's kappa
kappa = cohen_kappa_score(all_labels, all_predictions)
print(f"Cohen's Kappa: {kappa:.4f}")

# %%

