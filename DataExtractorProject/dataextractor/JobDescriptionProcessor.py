import json
import os.path
import pathlib
from DataExtractorProject.dataextractor.parsers.ParseJobDescToJson import ParseJobDesc

SAVE_DIRECTORY = "Data/Processed/JobDescription"


class JobDescriptionProcessor:
    def __init__(self, input_file, input_directory):
        """
        Initialize the processor with the input file and directory.
        :param input_file: The name of the job description file to process.
        :param input_directory: The directory containing the job description file.
        """
        self.input_file = input_file
        self.input_file_name = os.path.join(input_directory, self.input_file)

    def process(self) -> bool:
        """
        Processes the job description file and saves the processed JSON.
        :return: True if successful, False otherwise.
        """
        try:
            job_desc_dict = self._read_job_desc()
            self._write_json_file(job_desc_dict)
            return True
        except Exception as e:
            print(f"An error occurred while processing {self.input_file}: {str(e)}")
            return False

    def _read_job_desc(self) -> list:
        """
        Reads and parses the job description from the given file format.
        Supports JSON, CSV, and TXT formats.
        :return: A list of dictionaries containing the parsed job description data.
        """
        try:
            file_extension = os.path.splitext(self.input_file_name)[-1].lower()

            if file_extension == ".json":
                # Read JSON data
                with open(self.input_file_name, "r") as json_file:
                    data = json.load(json_file)
                parsed_data = [ParseJobDesc(row) for row in data]  # Iterate if multiple entries


            elif file_extension == ".csv":

                csv_data = self._read_csv_file(str(self.input_file_name))

                parsed_data = [ParseJobDesc(row) for row in csv_data]


            elif file_extension == ".txt":
                # Read plain text data
                with open(self.input_file_name, "r") as text_file:
                    data = text_file.read()
                parsed_data = [ParseJobDesc({"text": data})]

            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            return parsed_data

        except Exception as e:
            print(f"Error reading job description: {str(e)}")
            raise

    def _read_csv_file(self, file_path: str) -> list:
        """
        Reads and processes a CSV file containing job descriptions.
        Assumes the CSV file contains multiple rows, each representing a job description.
        :param file_path: The path to the CSV file.
        :return: A list of dictionaries, each representing a row in the CSV.
        """
        import csv
        with open(file_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)  # Convert to a list of dictionaries

        if not rows:
            raise ValueError("CSV file is empty.")

        return rows

    def _write_json_file(self, job_desc_dictionary: dict):
        """
        Writes the processed job description data to a JSON file.
        :param job_desc_dictionary: The dictionary containing the job description data.
        """
        file_name = (
            f"JobDescription-{self.input_file}-{job_desc_dictionary.get('unique_id', 'unknown')}.json"
        )
        save_directory_name = pathlib.Path(SAVE_DIRECTORY) / file_name
        json_object = json.dumps(job_desc_dictionary, sort_keys=True, indent=4)

        # Ensure the save directory exists
        pathlib.Path(SAVE_DIRECTORY).mkdir(parents=True, exist_ok=True)

        # Write the JSON object to the file
        with open(save_directory_name, "w+") as outfile:
            outfile.write(json_object)
