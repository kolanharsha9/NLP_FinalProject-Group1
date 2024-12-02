import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import random

# Constants
EMBEDDING_PATH = "data/processed/"
MODEL_SAVE_PATH = "data/models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingDataset(Dataset):
    """
    Custom Dataset to handle mismatched batches of resume and job description embeddings.
    """
    def __init__(self, resume_embeddings, job_desc_embeddings):
        self.resume_embeddings = resume_embeddings
        self.job_desc_embeddings = job_desc_embeddings
        self.resume_size = self.resume_embeddings.size(0)
        self.job_desc_size = self.job_desc_embeddings.size(0)

    def __len__(self):
        # Define length as the maximum of the two datasets
        return max(self.resume_size, self.job_desc_size)

    def __getitem__(self, idx):
        # For resume embeddings, cycle through if idx exceeds the size
        resume_idx = idx % self.resume_size
        # For job description embeddings, cycle through if idx exceeds the size
        job_desc_idx = idx % self.job_desc_size
        return self.resume_embeddings[resume_idx], self.job_desc_embeddings[job_desc_idx]


class MatchingModel(nn.Module):
    """
    Model to generate embeddings for resumes and job descriptions.
    """
    def __init__(self, embedding_dim):
        super(MatchingModel, self).__init__()
        self.fc_resume = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.fc_job = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, resume_embedding, job_desc_embedding):
        resume_output = self.fc_resume(resume_embedding)
        job_desc_output = self.fc_job(job_desc_embedding)
        return resume_output, job_desc_output


def train_model(embedding_dim=768, epochs=5, learning_rate=1e-3, num_job_desc_batches_per_resume=5):
    """
    Trains a neural network model using precomputed embeddings.
    """
    # Get the list of batch files
    resume_batch_files = [
        os.path.join(EMBEDDING_PATH, f)
        for f in os.listdir(EMBEDDING_PATH)
        if f.startswith('resume_batch') and f.endswith('.pt')
    ]
    job_desc_batch_files = [
        os.path.join(EMBEDDING_PATH, f)
        for f in os.listdir(EMBEDDING_PATH)
        if f.startswith('job_desc_batch') and f.endswith('.pt')
    ]

    resume_batch_files.sort()
    job_desc_batch_files.sort()

    # Initialize model, loss, and optimizer
    model = MatchingModel(embedding_dim).to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Shuffle job description batches at the start of each epoch
        random.shuffle(job_desc_batch_files)

        for resume_batch_file in tqdm(resume_batch_files, desc=f"Epoch {epoch + 1}/{epochs} - Resume Batches"):
            # Load resume embeddings
            resume_embeddings = torch.load(resume_batch_file, map_location='cpu')

            # Ensure embeddings are tensors
            if not isinstance(resume_embeddings, torch.Tensor):
                resume_embeddings = torch.tensor(resume_embeddings)

            # Do NOT move resume embeddings to DEVICE here

            # Randomly sample a subset of job description batches
            num_samples = min(num_job_desc_batches_per_resume, len(job_desc_batch_files))
            sampled_job_desc_batches = random.sample(job_desc_batch_files, num_samples)

            for job_desc_batch_file in sampled_job_desc_batches:
                # Load job description embeddings
                job_desc_embeddings = torch.load(job_desc_batch_file, map_location='cpu')

                # Ensure embeddings are tensors
                if not isinstance(job_desc_embeddings, torch.Tensor):
                    job_desc_embeddings = torch.tensor(job_desc_embeddings)

                # Do NOT move job description embeddings to DEVICE here

                # Create Dataset and DataLoader
                dataset = EmbeddingDataset(resume_embeddings, job_desc_embeddings)
                dataloader = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=True
                )

                for resume_emb_batch, job_desc_emb_batch in dataloader:
                    # Move batches to DEVICE
                    resume_emb_batch = resume_emb_batch.to(DEVICE)
                    job_desc_emb_batch = job_desc_emb_batch.to(DEVICE)

                    # Generate negative samples by shuffling job descriptions within the batch
                    permuted_indices = torch.randperm(job_desc_emb_batch.size(0))
                    negative_job_desc_emb = job_desc_emb_batch[permuted_indices]

                    # Forward pass
                    anchor_output, positive_output = model(resume_emb_batch, job_desc_emb_batch)
                    _, negative_output = model(resume_emb_batch, negative_job_desc_emb)

                    # Compute triplet loss
                    loss = criterion(anchor_output, positive_output, negative_output)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Clean up to free memory
                del job_desc_embeddings
                torch.cuda.empty_cache()

            # Clean up to free memory
            del resume_embeddings
            torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")

    # Save the trained model
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "matching_model.pt"))
    print("Model training complete. Model saved to:", MODEL_SAVE_PATH)
