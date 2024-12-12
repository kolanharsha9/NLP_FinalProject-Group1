import os
from src.pipeline import preprocess_pipeline
from src.model_training import train_model

if __name__ == "__main__":

    RESUME_DIR = "data/resumes"
    JOB_DESC_DIR = "data/job_descriptions"
    OUTPUT_DIR = "data/processed"
    EMBEDDING_DIR = "data/processed/"  # Directory for storing embeddings

    try:
        # Step 1: Preprocessing Pipeline
        print("Starting preprocessing pipeline...")
        preprocess_pipeline(
            resume_dir=RESUME_DIR,
            job_desc_dir=JOB_DESC_DIR,
            output_dir=OUTPUT_DIR,
            batch_size=100
        )
        print("Preprocessing completed.")

        # Step 2: Model Training
        print("Starting model training...")
        train_model(
            embedding_dim=768,  # Dimension of the embeddings
            epochs=5,           # Number of epochs
            learning_rate=1e-3  # Learning rate for training
        )
        print("Model training completed successfully.")

    except Exception as e:
        print(f"Pipeline error: {e}")
