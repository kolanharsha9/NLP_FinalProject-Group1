def preprocess_resume(resume_data):
    """Extract and concatenate relevant sections from a resume."""
    try:
        clean_data = resume_data.get("clean_data", "")
        extracted_keywords = " ".join(resume_data.get("extracted_keywords", []))
        return f"{clean_data} {extracted_keywords}"
    except Exception as e:
        print(f"Error preprocessing resume: {e}")
        return ""
