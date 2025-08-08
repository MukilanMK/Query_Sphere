import streamlit as st
from sentence_transformers import SentenceTransformer
import json
import fitz  # PyMuPDF
import faiss
import numpy as np
import io
import email
from email.policy import default
import docx  # For .docx files
from groq import Groq
import requests  # For fetching files from URLs

# --- Backend Functions ---

def fetch_file_from_url(url: str) -> bytes | None:
    """Downloads a file's content from a public URL."""
    try:
        # Some servers block requests without a user-agent
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        # Raises an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch file from URL: {url}\nDetails: {e}")
        return None


def extract_text_from_document(file_bytes: bytes, file_name: str) -> list[dict] | None:
    """Extracts text chunks from a file's bytes based on its name."""
    text_chunks = []
    try:
        if file_name.lower().endswith('.pdf'):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4].replace('\n', ' ').strip()
                    if len(text) > 40:
                        text_chunks.append(
                            {"text": text, "page": page_num + 1})

        elif file_name.lower().endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text = para.text.strip()
                if len(text) > 40:
                    text_chunks.append({"text": text, "page": "N/A"})

        elif file_name.lower().endswith('.eml'):
            msg = email.message_from_bytes(file_bytes, policy=default)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == 'text/plain' and 'attachment' not in content_disposition:
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                if msg.get_content_type() == 'text/plain':
                    body = msg.get_payload(decode=True).decode()

            for para in body.split('\n\n'):
                text = para.replace('\n', ' ').strip()
                if len(text) > 40:
                    text_chunks.append({"text": text, "page": "N/A"})

        return text_chunks if text_chunks else None
    except Exception as e:
        print(f"ERROR: Failed to process file {file_name}.\nDetails: {e}")
        return None


def create_faiss_index(text_chunks: list[str], embedding_model):
    """Creates a FAISS index using a loaded SentenceTransformer model."""
    try:
        embeddings = embedding_model.encode(
            text_chunks, show_progress_bar=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index
    except Exception as e:
        print(f"ERROR: Failed to create local embeddings.\nDetails: {e}")
        return None


def synthesize_answer_with_groq(client, query: str, retrieved_clauses: list[dict]):
    """Generates a synthesized answer using Llama 3 on Groq."""
    context = "\n\n".join(
        [f"Clause (from page {c['page']}): {c['text']}" for c in retrieved_clauses])
    system_prompt = "You are a helpful assistant. Your response MUST be a valid JSON object with three keys: 'relevant_clause', 'explanation', and 'page_number'. If the answer cannot be found, state that in the 'explanation' field."
    user_prompt = f"Based ONLY on the CONTEXT below, answer the user's QUERY.\n\nCONTEXT:\n{context}\n\nQUERY: {query}"
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(
            f"ERROR: Failed to generate answer with Groq/Llama 3.\nDetails: {e}")
        return None


def process_query(api_key: str, input_source, input_type: str, query: str, embedding_model) -> dict:
    """Main processing pipeline that handles either a file upload or a URL."""
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return {"error": f"Failed to initialize Groq client. Details: {e}"}

    file_bytes = None
    file_name = ""

    if input_type == 'upload':
        file_bytes = input_source.getvalue()
        file_name = input_source.name
    elif input_type == 'url':
        file_bytes = fetch_file_from_url(input_source)
        if not file_bytes:
            return {"error": "Failed to download the file from the URL. Please check if the URL is correct and public."}
        # Infer filename from URL
        file_name = input_source.split('/')[-1].split('?')[0]

    document_chunks_with_metadata = extract_text_from_document(
        file_bytes, file_name)
    if not document_chunks_with_metadata:
        return {"error": "Could not extract any readable text from the document."}

    document_texts = [chunk['text']
                      for chunk in document_chunks_with_metadata]
    index = create_faiss_index(document_texts, embedding_model)
    if not index:
        return {"error": "Failed to create document embeddings."}

    query_embedding = embedding_model.encode([query])
    k = 3
    distances, indices = index.search(
        np.array(query_embedding).astype('float32'), k)
    retrieved_clauses = [document_chunks_with_metadata[i]
                         for i in indices[0]]

    synthesized_answer_str = synthesize_answer_with_groq(
        client, query, retrieved_clauses)
    if not synthesized_answer_str:
        return {"error": "The AI model failed to generate a response."}

    try:
        return json.loads(synthesized_answer_str)
    except json.JSONDecodeError:
        return {"error": "The AI model failed to return valid JSON."}


# --- Streamlit UI ---

# --- API Key Config ---
# IMPORTANT: Replace the placeholder with your actual key.
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- Page Setup ---
st.set_page_config(
    page_title="querysphere",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .stApp {
            background-image: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
            color: #F9FAFB;
        }
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF; font-weight: 700; }
        h1 { text-align: center; text-shadow: 2px 2px 8px rgba(0,0,0,0.3); }
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] {
            background-color: rgba(31, 41, 55, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 0.75rem;
            padding: 2rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        }
        .stButton > button {
            background-color: #FFFFFF;
            color: #1F2937;
            font-weight: 600;
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover { background-color: #E5E7EB; transform: scale(1.02); }
    </style>
""", unsafe_allow_html=True)


# --- Session State & Model Loading ---
if "result" not in st.session_state:
    st.session_state.result = None


@st.cache_resource
def load_embedding_model():
    """Loads and caches the embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


# --- Header ---
st.title("🤖 DocuQuery AI")
st.markdown("<p style='text-align: center; color: #D1D5DB;'>Upload a document or provide a URL, ask a question, and get an AI-powered JSON response.</p>", unsafe_allow_html=True)

# --- Model Loading ---
with st.spinner("Initializing AI model..."):
    embedding_model = load_embedding_model()

# --- Two-Column Layout ---
col1, col2 = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: INPUTS ---
with col1:
    st.header("Controls")
    input_method = st.radio(
        "Choose document source:",
        ("Upload a File", "From a URL"),
        horizontal=True
    )

    uploaded_file = None
    url_input = ""

    if input_method == "Upload a File":
        uploaded_file = st.file_uploader(
            "Select a document", type=["pdf", "docx", "eml"])
    else:
        url_input = st.text_input(
            "Enter document URL", placeholder="https://example.com/document.pdf")

    query = st.text_input("Enter your question",
                          placeholder="e.g., What are the key conclusions?")
    submit_button = st.button("Generate Response")

# --- RIGHT COLUMN: OUTPUTS ---
with col2:
    st.header("Result")

    if submit_button:
        if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
            st.error("⚠️ Please add your Groq API key to the `app.py` script.")
        elif not query:
            st.error("⚠️ Please enter a question.")
        else:
            input_source = None
            input_type = ""
            is_valid_input = False

            if input_method == "Upload a File":
                if uploaded_file:
                    input_source = uploaded_file
                    input_type = "upload"
                    is_valid_input = True
                else:
                    st.error("⚠️ Please upload a document.")
            else:  # From URL
                if url_input:
                    input_source = url_input
                    input_type = "url"
                    is_valid_input = True
                else:
                    st.error("⚠️ Please enter a URL.")

            if is_valid_input:
                with st.spinner("🤖 Analyzing document and generating answer..."):
                    st.session_state.result = process_query(
                        GROQ_API_KEY,
                        input_source,
                        input_type,
                        query,
                        embedding_model
                    )

    if st.session_state.result:
        result = st.session_state.result
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            st.success("✅ Query processed successfully!")
            st.json(result)
    else:
        st.info("The JSON result will appear here once you submit a query.")