def preprocess_job_desc(job_data):
    """Extract and concatenate relevant sections from a job description."""
    try:
        clean_data = job_data.get("clean_data", "")
        extracted_keywords = " ".join(job_data.get("extracted_keywords", []))

        entities = " ".join(job_data.get("entities", []))

        job_desc_details = ""
        if "job_desc_data" in job_data:
            import ast
            try:
                job_desc_dict = ast.literal_eval(job_data["job_desc_data"])
                job_desc_details = " ".join(str(value) for value in job_desc_dict.values())
            except Exception as e:
                print(f"Error parsing job_desc_data: {e}")

        combined_text = f"{clean_data} {extracted_keywords} {entities} {job_desc_details}"

        return combined_text.strip()

    except Exception as e:
        print(f"Error preprocessing job description: {e}")
        return ""
