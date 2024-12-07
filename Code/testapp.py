import streamlit as st
from PIL import Image
import base64
import requests
from io import BytesIO
from models.ModelForGrammaticalAndFormating.model import BedrockResumeAnalyzer

# Set page layout
st.set_page_config(page_title="Chat Interface Demo", page_icon="💬", layout="wide")

# AWS credentials (Replace these with secure methods in production)
AWS_ACCESS_KEY_ID = "ASIATP2NJQDCZCXPCZQ5"
AWS_SECRET_ACCESS_KEY = "h4fuP2jTl7InNM0lL9IK0qT9uJj58cOqR3ZpSyxL"
AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEHwaCXVzLWVhc3QtMSJGMEQCIFWDMG4TXGnMUrjxwK1ygEfQGt20VUnRAsx8anbiKCrTAiBQbAQ37qFtG6HQhEKIjN8oP/7olxvaP7Rrjtn+1dgK+iqaAwg1EAEaDDI0MDE0MzQwMTE1NyIMIJT2wYrwPPsY/q9sKvcCnZRRjcXi3me1TALvOAkz1l5YfU8aCsz0dcdXFaJ0M92nVFK62mwNzkoGofxEDxH6tFQXGS8KIN3aVMuXpurED1y4xITipLkY3h3xfH2zPACrvEhPCw8yFxHTFtk0zN0QQaXBIxx3Y8FQmfqUu7cFJbvzlpwz9NtNm0SRc8YzP5lSQ9AigPZdXG1FzRQu71kXw+rO0g8dciqF+vRASLzZhM8Zf82ndND7DdzqSH90A+mwf9Sf+PIP11m+yPRMA4/E6peqg+pwmaNHLJq4BcwmW3fGPBtlDummxa4fG2xVAnj2uUzNkhDlAAVrQsUJITBP/GczvzmJUiFgeC1YGtYSTcMXdrE3tqaRbIdvnws5zb4CJlvtzJUCQQ7cnLkihr/Yo4/aDiEkedhz3pvc3nqLlD7p/T+2ggnRjAEBvT0lJylef+ZPy40xj/PhzVdH9xKfbMKg+4lX8kvBsa12k14d0PB/oKxHavf9WPQpfA+u0lkAUiM0V4ZzMLahzboGOqcBZ/HWj2QgL+aFrr/CS6VTEbRD+s5190KH3iEbTanSSwvwGkbhAye5jjgCZijqfMBDf/sRg1By9bPN549TaDe/nCLgXfaZmPOYII6yDfDNgFttESVha3rATmPhQQl19fAXx8rF3FNPSvBkgbZIdvjikzMcOGlTyPwxwJ3UtSsjvZeaNE7xCNIQat+MphNKf/igW3HjW8i0RoBGviucwyxCWDrb8iEHY1s="
REGION_NAME = "us-east-1"


@st.cache_resource
def initialize_analyzer():
    return BedrockResumeAnalyzer(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=REGION_NAME,
    )


def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


analyzer = initialize_analyzer()
attachment_icon = get_base64_image("png/png1.png")

# --- CSS for Styling ---
st.markdown(f"""
<style>
body {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}
.chat-container {{
    max-width: 800px;
    width: 100%;
    margin: 0 auto;
    padding: 1em;
    background-color: #F9F9F9;
    border-radius: 10px;
}}
.user-message {{
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    max-width: 80%;
    text-align: left;
}}
.bot-message {{
    background-color: #E2E2E2;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    max-width: 80%;
    text-align: left;
}}
.message-container {{
    display: flex;
    flex-direction: column;
}}
.user-bubble {{
    align-self: flex-end;
}}
.bot-bubble {{
    align-self: flex-start;
}}
.model-selection {{
    margin-bottom: 20px;
}}

/* Center the title and content */
h1, h2, h3, h4, h5, h6, p, div, .stMarkdown {{
    text-align: center;
}}

/* Make the file uploader look like a small attachment icon */
div[data-testid="stFileUploadDropzone"] {{
    background: url("data:image/png;base64,{attachment_icon}") no-repeat center center;
    background-size: 24px 24px;  /* Adjust size as needed */
    border: none !important;
    height: 30px !important;
    width: 30px !important;
    cursor: pointer;
    margin: 0 !important;
}}
div[data-testid="stFileUploadDropzone"]::before {{
    content: none !important; /* Hide default content */
}}
div[data-testid="stFileUploadDropzone"] > div {{
    display: none !important; /* Hide default instructions */
}}

/* Align the chat input row center */
.chat-input-row {{
    max-width: 800px;
    width: 100%;
    margin: 1em auto 0 auto;
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    gap: 0.5em;
}}

/* Narrow text input box */
.chat-text-input {{
    flex: 1;
    margin: 0 !important;
}}

/* Submit button inline */
.chat-submit-btn {{
    margin: 0 !important;
}}
</style>
""", unsafe_allow_html=True)


def resume_generation_page():
    st.title("Resume Generation")
    st.write("Generate a professional resume using our AI-powered tool.")
    user_prompt = st.text_area("Enter your details (e.g., name, experience, skills):", key="resume_gen_prompt")

    generate_button = st.button("Generate Resume")

    if generate_button and user_prompt.strip():
        with st.spinner("Generating your resume..."):
            generated_resume = "Generated resume text here"
            st.success("Resume generated successfully!")
            st.text_area("Generated Resume", generated_resume, height=300)
            st.download_button("Download Resume", generated_resume, file_name="generated_resume.txt", mime="text/plain")


def presentation_page():
    st.title("Project Presentation")
    st.write("Below, you can present images, charts, or other visual media alongside explanatory text.")

    # Example: Display an image from URL or local file
    example_image_url = "https://via.placeholder.com/400"
    try:
        response = requests.get(example_image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Example Project Image")
    except:
        st.write("Unable to load example image.")

    st.write("""
    **Project Description:**

    This section can contain detailed explanations, notes, and additional writings about your project.

    - Show multiple images.
    - Integrate tables or data visualizations.
    - Add expander sections with additional details.
    - Embed videos or audio clips.
    """)


def main_page():
    st.sidebar.title("Model Settings")
    model_option = st.sidebar.selectbox(
        "Choose a model:",
        ["GF-0.1", "GPT-4", "Local LLM (e.g., Llama 2)", "HuggingFace Model"]
    )

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("Multi-Model Chat Interface")
    st.write("Interact with different language models and experience a GPT-like interface.")

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f'<div class="message-container user-bubble"><div class="user-message">{content}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-container bot-bubble"><div class="bot-message">{content}</div></div>',
                        unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col_input, col_upload = st.columns([4, 1], gap="small")

    with col_input:
        user_input = st.text_input("Type your message:")

    with col_upload:
        uploaded_file = None
        if model_option == "GF-0.1":
            uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    send_button = st.button("Send")

    if send_button and user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})

        if model_option == "GF-0.1" and uploaded_file is not None:
            with st.spinner("Processing your resume..."):
                st.info("Extracting text from your resume...")
                resume_text = "Extracted resume text here"

                st.info("Analyzing resume for grammar and formatting...")
                analysis_results = {
                    "grammar_score": 95,
                    "formatting_score": 90,
                    "grammatical_errors": ["Error 1", "Error 2"],
                    "formatting_issues": ["Issue 1"],
                    "recommendations": ["Fix 1", "Fix 2"]
                }

            analysis_summary = f"""
**Grammar Score:** {analysis_results['grammar_score']}/100  
**Formatting Score:** {analysis_results['formatting_score']}/100  

**Grammatical Errors:**  
{', '.join(analysis_results['grammatical_errors'])}

**Formatting Issues:**  
{', '.join(analysis_results['formatting_issues'])}

**Recommendations:**  
{', '.join(analysis_results['recommendations'])}
"""
            st.session_state["messages"].append({"role": "assistant", "content": analysis_summary})
        else:
            st.session_state["messages"].append({"role": "assistant", "content": "Placeholder response"})

        st.experimental_rerun()


tabs = st.tabs(["Main Page", "Presentation", "Resume Generation"])

with tabs[0]:
    main_page()
with tabs[1]:
    presentation_page()
with tabs[2]:
    resume_generation_page()
