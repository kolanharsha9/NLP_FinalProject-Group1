import json
import os
import pathlib

from DataExtractorProject.dataextractor.Extractor import DataExtractor
from DataExtractorProject.dataextractor.utils.Utils import TextCleaner, generate_unique_id

SAVE_DIRECTORY = "../../Data/Processed/JobDescription"


class ParseJobDesc:
    def __init__(self, job_desc):
        if isinstance(job_desc, dict):
            job_desc_text = job_desc.get('text', str(job_desc))
        else:
            job_desc_text = str(job_desc)

        self.job_desc_data = job_desc_text
        self.clean_data = TextCleaner.clean_text(self.job_desc_data)
        self.entities = DataExtractor(self.clean_data).extract_entities()
        self.key_words = DataExtractor(self.clean_data).extract_particular_words()

    def get_JSON(self) -> dict:
        """
        Returns a dictionary of job description data.
        """
        job_desc_dictionary = {
            "unique_id": generate_unique_id(),
            "job_desc_data": self.job_desc_data,
            "clean_data": self.clean_data,
            "entities": self.entities,
            "extracted_keywords": self.key_words,
        }

        return job_desc_dictionary