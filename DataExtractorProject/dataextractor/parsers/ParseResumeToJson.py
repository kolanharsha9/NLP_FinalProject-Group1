import json
import os
from datetime import datetime
from pathlib import Path

from DataExtractorProject.dataextractor.Extractor import DataExtractor
from DataExtractorProject.dataextractor.utils.Utils import TextCleaner, generate_unique_id


SAVE_DIRECTORY = os.getenv("RESUME_SAVE_DIRECTORY", "../../Data/Processed/Resumes")
Path(SAVE_DIRECTORY).mkdir(parents=True, exist_ok=True)


class ParseResume:
    def __init__(self, resume: str):
        self.resume_data = resume
        self.clean_data = TextCleaner.clean_text(self.resume_data)
        self.extractor = DataExtractor(self.clean_data)

        # Extracting necessary Data
        self.entities = self.extractor.extract_entities()
        self.experience = self.extractor.extract_experience()
        self.key_words = self.extractor.extract_particular_words()

    def get_JSON(self) -> dict:
        """
        Returns a dictionary of resume Data with metadata.
        """
        try:
            resume_dictionary = {
                "unique_id": generate_unique_id(),
                "resume_data": self.resume_data,
                "clean_data": self.clean_data,
                "entities": self.entities,
                "extracted_keywords": self.key_words,
                "experience": self.experience,
                "processed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"Error generating JSON: {e}")
            resume_dictionary = {
                "error": f"Failed to process resume Data: {str(e)}"
            }
        return resume_dictionary
