import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from models.ModelForGrammaticalAndFormating.model import BedrockResumeAnalyzer

# Set page layout
st.set_page_config(page_title="Chat Interface Demo", page_icon="💬", layout="wide")

# AWS credentials (Replace with actual credentials or load from environment)
AWS_ACCESS_KEY_ID = "ASIATP2NJQDCZCXPCZQ5"
AWS_SECRET_ACCESS_KEY = "h4fuP2jTl7InNM0lL9IK0qT9uJj58cOqR3ZpSyxL"
AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEHwaCXVzLWVhc3QtMSJGMEQCIFWDMG4TXGnMUrjxwK1ygEfQGt20VUnRAsx8anbiKCrTAiBQbAQ37qFtG6HQhEKIjN8oP/7olxvaP7Rrjtn+1dgK+iqaAwg1EAEaDDI0MDE0MzQwMTE1NyIMIJT2wYrwPPsY/q9sKvcCnZRRjcXi3me1TALvOAkz1l5YfU8aCsz0dcdXFaJ0M92nVFK62mwNzkoGofxEDxH6tFQXGS8KIN3aVMuXpurED1y4xITipLkY3h3xfH2zPACrvEhPCw8yFxHTFtk0zN0QQaXBIxx3Y8FQmfqUu7cFJbvzlpwz9NtNm0SRc8YzP5lSQ9AigPZdXG1FzRQu71kXw+rO0g8dciqF+vRASLzZhM8Zf82ndND7DdzqSH90A+mwf9Sf+PIP11m+yPRMA4/E6peqg+pwmaNHLJq4BcwmW3fGPBtlDummxa4fG2xVAnj2uUzNkhDlAAVrQsUJITBP/GczvzmJUiFgeC1YGtYSTcMXdrE3tqaRbIdvnws5zb4CJlvtzJUCQQ7cnLkihr/Yo4/aDiEkedhz3pvc3nqLlD7p/T+2ggnRjAEBvT0lJylef+ZPy40xj/PhzVdH9xKfbMKg+4lX8kvBsa12k14d0PB/oKxHavf9WPQpfA+u0lkAUiM0V4ZzMLahzboGOqcBZ/HWj2QgL+aFrr/CS6VTEbRD+s5190KH3iEbTanSSwvwGkbhAye5jjgCZijqfMBDf/sRg1By9bPN549TaDe/nCLgXfaZmPOYII6yDfDNgFttESVha3rATmPhQQl19fAXx8rF3FNPSvBkgbZIdvjikzMcOGlTyPwxwJ3UtSsjvZeaNE7xCNIQat+MphNKf/igW3HjW8i0RoBGviucwyxCWDrb8iEHY1s="
REGION_NAME = "us-east-1"

# Initialize the analyzer for GF-0.1 model usage
@st.cache_resource
def initialize_analyzer():
    return BedrockResumeAnalyzer(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=REGION_NAME,
    )

analyzer = initialize_analyzer()

# --- Styling ---
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 1em;
    background-color: #F9F9F9;
    border-radius: 10px;
}

.user-message {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: left;
    max-width: 80%;
}

.bot-message {
    background-color: #E2E2E2;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: left;
    max-width: 80%;
}

.message-container {
    display: flex;
    flex-direction: column;
}

.user-bubble {
    align-self: flex-end;
}

.bot-bubble {
    align-self: flex-start;
}

.model-selection {
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar for Model Selection ---
st.sidebar.title("Model Settings")
model_option = st.sidebar.selectbox(
    "Choose a model:",
    ["GF-0.1", "OpenAI GPT-4", "Local LLM (e.g., Llama 2)", "HuggingFace Model"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
st.sidebar.write("**Note:** Integrate your model’s API keys or endpoints in the code to use the chosen model.")

# --- Initialize Session State for chat history ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Header / Title ---
st.title("Multi-Model Chat Interface")
st.write("Interact with different language models, display images, and present your project content.")

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

# --- User Input and File Upload in Chat Section ---
col_input, col_upload = st.columns([4,1], gap="small")

with col_input:
    user_input = st.text_input("Type your message:")

with col_upload:
    # Show file uploader only if GF-0.1 is selected, as requested
    uploaded_file = None
    if model_option == "GF-0.1":
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

send_button = st.button("Send")

if send_button and user_input.strip():
    # Add user's message to the session
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Check if we are using GF-0.1 and a file is uploaded
    if model_option == "GF-0.1" and uploaded_file is not None:
        # Process the resume with a spinner and status messages
        with st.spinner("Processing your resume..."):
            st.info("Extracting text from your resume...")
            # Save the uploaded file
            with open("uploaded_resume.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text
            resume_text = analyzer.extract_text_from_pdf("uploaded_resume.pdf")

            st.info("Analyzing resume for grammar and formatting...")
            analysis_results = analyzer.analyze_resume_text(resume_text)

        # Format the analysis results
        analysis_summary = f"""
**Grammar Score:** {analysis_results.get('grammar_score', 'N/A')}/100  
**Formatting Score:** {analysis_results.get('formatting_score', 'N/A')}/100  

**Grammatical Errors:**  
{', '.join([error['description'] for error in analysis_results.get('grammatical_errors', [])]) or 'None'}

**Formatting Issues:**  
{', '.join([issue['description'] for issue in analysis_results.get('formatting_issues', [])]) or 'None'}

**Recommendations:**  
{', '.join([rec['description'] for rec in analysis_results.get('recommendations', [])]) or 'None'}
"""

        # Add the analysis result to the chat
        st.session_state["messages"].append({"role": "assistant", "content": analysis_summary})

    else:
        # If not GF-0.1 or no resume uploaded, just return a placeholder response
        mock_response = f"You said: {user_input}. (This is a placeholder response from {model_option}.)"
        st.session_state["messages"].append({"role": "assistant", "content": mock_response})

    # Rerun to display updated chat
    st.experimental_rerun()

# --- Project Presentation Section ---
st.subheader("Project Presentation")
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