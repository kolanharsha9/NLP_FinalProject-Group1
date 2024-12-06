def preprocess_job_desc(job_data):
    """Extract and concatenate relevant sections from a job description."""
    try:
        # Extract 'clean_data'
        clean_data = job_data.get("clean_data", "")

        # Extract and join 'extracted_keywords'
        extracted_keywords = " ".join(job_data.get("extracted_keywords", []))

        # Optionally, extract and join 'entities'
        entities = " ".join(job_data.get("entities", []))

        # Optionally, parse 'job_desc_data' if needed
        job_desc_details = ""
        if "job_desc_data" in job_data:
            # Convert the string representation of the dictionary to an actual dictionary
            import ast
            try:
                job_desc_dict = ast.literal_eval(job_data["job_desc_data"])
                # Extract specific fields if necessary
                job_desc_details = " ".join(str(value) for value in job_desc_dict.values())
            except Exception as e:
                print(f"Error parsing job_desc_data: {e}")

        # Concatenate all extracted information
        combined_text = f"{clean_data} {extracted_keywords} {entities} {job_desc_details}"

        # Return the combined text
        return combined_text.strip()

    except Exception as e:
        print(f"Error preprocessing job description: {e}")
        return ""
