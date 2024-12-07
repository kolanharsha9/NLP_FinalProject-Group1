import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import base64
from models.ModelForGrammaticalAndFormating.model import BedrockResumeAnalyzer
from models.resume_train_preprocess_test import gen_resume

# Set page layout
st.set_page_config(page_title="Chat Interface Demo", page_icon="💬", layout="wide")

# AWS credentials (Replace these with secure methods in production)
AWS_ACCESS_KEY_ID = "ASIATP2NJQDCYS4B2AR6"
AWS_SECRET_ACCESS_KEY = "t+xAk085Nnm5Mb7SteZyZIccqv5wUl+hUyjDsnMH"
AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEJT//////////wEaCXVzLWVhc3QtMSJHMEUCIQCyhFs2NB8H8vC0n25TjJDRCucyaSrITZK+b6TrkDcSpgIgIyY1O6+4ltQUwMoeUQzbEygDH67ye80F3gXMprQ0StQqmgMITRABGgwyNDAxNDM0MDExNTciDG2pGLzlqEu4cAnjQSr3AuOgmGQ1XgHd5c6MMlptMdV5ggX76b7kTonkNmmOblG8gQaW0KSdNMHqh82FvGLc05L8vRIdOKeN0djeNe/KREkZnxScMT3bg7o/Pzg8HxL/5WkkaU3EjWQKUbtNcOGpfYUqyBwJYnPDDOc2tW56Z2KJZWDdUbh+OVfTz/I7DqfTDJ1CbljkQezhBvHRB6bHmT8VpK/idZXWgii8ksxAWbyNlR7rebds+VEnjcc509J8riTeXQhBj7FQUYFRXlmGmE2XA4E9ORVfY61pD9rKqSBY/VBL/AKForTmpM/my5fwZt9ZWREj7jjJt96Gfakx+lfmqKmQT4oTGM7rsaCvn1AvF3GoBWcuLNQe0lI8vMu460CsDfjdMo1EF1Gj0HHTD88ribWC5PGSoh0iGnAUl1JqkwHa+Q72X7VsNUZQorZI3j2LL+Fl0Auitid8uIV8qfksVopmPf90gPkAV86gDJZhAagArkYRd5TnC7FBYbAyLqk752c6mDCAyNK6BjqmAdLA/HDqkjou0PQg8XAT2sh+y9Oj++j5YVX1BxfuApABsf1tDGDrPyWyPAq5OPM849b8SonsbrkvgEaKGJpPLYGbWlR+PHeNw6sHcE042AJ/7ZImjdQIhNhyH6ztvhjf1XNRa6kT8DO73YFbjW5e1hF3U4Ldmvo+35Z0MCPM1+jji8IN/k/eoCmcCu2TpB1spsOlXNOm1Zmtfgno4oxs0MgCyuq3kIc="
REGION_NAME = 'us-east-1'

@st.cache_resource
def initialize_analyzer():
    return BedrockResumeAnalyzer(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=REGION_NAME,
    )

analyzer = initialize_analyzer()

resume_generator = gen_resume()

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Encode your PNG attachment icon
attachment_icon = get_base64_image("png/png1.png")  # Update the path as necessary

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
    background-size: 60%;
    border: none !important;
    height: 30px !important;
    width: 30px !important;
    cursor: pointer;
    padding: 0 !important;
    margin: 0 !important;
}}
div[data-testid="stFileUploadDropzone"] > div {{
    display: none !important;
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
            generated_resume = resume_generator.generate_resume(user_prompt, resume_generator.model, resume_generator.tokenizer)
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

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://via.placeholder.com/200?text=Image+1", caption="Conceptual Diagram")
    with col2:
        st.image("https://via.placeholder.com/200?text=Image+2", caption="Model Architecture")

    st.write("Feel free to customize this section to best represent your project and its contents.")


def main_page():
    # --- Sidebar for Model Selection ---
    st.sidebar.title("Model Settings")
    model_option = st.sidebar.selectbox(
        "Choose a model:",
        ["GF-0.1", "GPT-4", "Local LLM (e.g., Llama 2)", "HuggingFace Model"]
    )

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    st.sidebar.write("**Note:** Integrate your model’s API keys or endpoints in the code to use the chosen model.")

    # --- Initialize Session State for chat history ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # --- Title and Introduction in the center ---
    st.markdown("<h1>Multi-Model Chat Interface</h1>", unsafe_allow_html=True)
    st.write("Interact with different language models and experience a GPT-like interface.")

    # --- Main Chat Container ---
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display the conversation
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
    if "trigger_rerun" not in st.session_state:
        st.session_state["trigger_rerun"] = False

    # --- Message Input Row ---
    st.markdown('<div class="chat-input-row">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap="small")

    # File uploader for PDF resumes
    with c1:
        uploaded_file = st.file_uploader(
            "Upload PDF",  # Accessible label (not visible)
            type="pdf",
            label_visibility="collapsed",
            key="chat_file"
        )

    # Text input for user messages
    with c2:
        user_input = st.text_input(
            "Chat Input",  # Accessible label (not visible)
            placeholder="Type your message here...",
            label_visibility="collapsed",
            key="chat_input",
            help="Type your message and press the send button"
        )

    # Send button
    with c3:
        send_button = st.button("➡️", key="send_button", help="Send message")

    st.markdown('</div>', unsafe_allow_html=True)

    if send_button:
        if uploaded_file:
            with st.spinner("Processing your resume..."):
                st.info("Extracting text from your resume...")

                # Save the uploaded file locally
                with open("uploaded_resume.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract text from the uploaded resume
                resume_text = analyzer.extract_text_from_pdf("uploaded_resume.pdf")

                st.info("Analyzing resume for grammar and formatting...")

                # Analyze the resume
                analysis_results = analyzer.analyze_resume_text(resume_text)

                # Format analysis results
                analysis_summary = f"""
    **Grammar Score:** {analysis_results.get('grammar_score', 'N/A')}/100  
    **Formatting Score:** {analysis_results.get('formatting_score', 'N/A')}/100  

    **Grammatical Errors:**  
    {', '.join([err['description'] for err in analysis_results.get('grammatical_errors', [])]) or 'None'}

    **Formatting Issues:**  
    {', '.join([issue['description'] for issue in analysis_results.get('formatting_issues', [])]) or 'None'}

    **Recommendations:**  
    {', '.join([rec['description'] for rec in analysis_results.get('recommendations', [])]) or 'None'}
    """
                # Add the analysis result to the chat
                st.session_state["messages"].append({"role": "assistant", "content": analysis_summary})

        # Display the chat messages
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="message-container user-bubble"><div class="user-message">{msg["content"]}</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="message-container bot-bubble"><div class="bot-message">{msg["content"]}</div></div>',
                    unsafe_allow_html=True)

        # If a user message is provided
        if user_input.strip():
            # Add user's message to the chat session
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # Mock response for non-GF-0.1 models
            if model_option != "GF-0.1":
                mock_response = f"You said: {user_input}. (This is a placeholder response from {model_option}.)"
                st.session_state["messages"].append({"role": "assistant", "content": mock_response})

        # Handle the case where no input is provided
        if not uploaded_file and not user_input.strip():
            st.error("Please upload a PDF file or type a message.")

        # Clear the input dynamically (re-render the text input widget)
        st.session_state["trigger_rerun"] = not st.session_state["trigger_rerun"]


# Tabs: Main Chat, Presentation, and Resume Generation
tabs = st.tabs(["Main Page", "Presentation", "Resume Generation"])

with tabs[0]:
    main_page()
with tabs[1]:
    presentation_page()
with tabs[2]:
    resume_generation_page()
