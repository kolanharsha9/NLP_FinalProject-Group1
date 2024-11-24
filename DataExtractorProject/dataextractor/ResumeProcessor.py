import json
from pathlib import Path
from datasets import load_dataset
from dataextractor.parsers.ParseResumeToJson import ParseResume  # Ensure this path is correct

# Directory to save processed JSON files
SAVE_DIRECTORY = "Data/Processed/Resumes"

class ResumeProcessor:
    def __init__(self, dataset_name: str):
        """
        Initialize the ResumeProcessor with the dataset name.
        :param dataset_name: Name of the Hugging Face dataset to process.
        """
        self.dataset_name = dataset_name
        self.dataset = load_dataset(self.dataset_name)

    def process(self) -> bool:
        """
        Process the dataset by parsing resumes and saving the results as JSON files.
        """
        try:
            # Access the 'train' split directly
            records = self.dataset["train"]  # Explicitly access the data split
            print(f"Dataset contains {len(records)} records.")  # Debugging

            for record in records:
                try:
                    # Ensure the record has "Resume_test" before processing
                    if "Resume_test" in record and record["Resume_test"]:
                        resume_dict = self._parse_resume(record)
                        self._write_json_file(resume_dict, record.get("instruction", "Unknown"))
                        print(f"File written for record: {record.get('instruction', 'Unknown')}")  # Debugging
                    else:
                        print(f"Skipped record due to missing or empty 'Resume_test': {record}")  # Debugging
                except Exception as inner_e:
                    print(f"Error processing record: {record}. Error: {inner_e}")
            return True
        except Exception as e:
            print(f"An error occurred during processing: {str(e)}")
            return False

    def _parse_resume(self, record: dict) -> dict:
        """
        Parse a resume record and extract relevant details.
        :param record: A dictionary containing the resume text and metadata.
        :return: A dictionary of parsed resume details.
        """
        resume_text = record["Resume_test"]
        parser = ParseResume(resume_text)
        output = parser.get_JSON()
        output.update({
            "instruction": record.get("Instruction", ""),
            "input_field": record.get("input", None),
        })
        return output

    def _write_json_file(self, resume_dictionary: dict, instruction: str):
        """
        Save the processed resume data as a JSON file.
        :param resume_dictionary: Dictionary of processed resume data to save.
        :param instruction: Instruction field for categorizing files.
        """
        file_name = f"Resume-{instruction}-{resume_dictionary['unique_id']}.json"
        save_directory_name = Path(SAVE_DIRECTORY) / file_name
        json_object = json.dumps(resume_dictionary, sort_keys=True, indent=4)
        save_directory_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(save_directory_name, "w") as outfile:
            outfile.write(json_object)
