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
                    if len(text) > 40: # Filter out very short text blocks
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


def process_query(api_key: str, uploaded_file, query: str, embedding_model) -> dict:
    """Main processing pipeline for an uploaded file."""
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return {"error": f"Failed to initialize Groq client. Details: {e}"}

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

    document_chunks_with_metadata = extract_text_from_document(
        file_bytes, file_name)
    if not document_chunks_with_metadata:
        return {"error": "Could not extract any readable text from the document."}

    document_texts = [chunk['text']
                      for chunk in document_chunks_with_metadata]
    index = create_faiss_index(document_texts, embedding_model)
    if index is None:
        return {"error": "Failed to create document embeddings."}

    query_embedding = embedding_model.encode([query])
    k = 3  # Retrieve top 3 most relevant chunks
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
# For deployment, it's recommended to use st.secrets for your API key.
# Create a file .streamlit/secrets.toml and add:
# GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE" # Fallback for local testing

# --- Page Setup ---
st.set_page_config(
    page_title="Query Sphere",
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
st.title("🤖 Query Sphere")
st.markdown("<p style='text-align: center; color: #D1D5DB;'>Upload a document, ask a question, and get an AI-powered response.</p>", unsafe_allow_html=True)

# --- Model Loading ---
with st.spinner("Initializing AI model..."):
    embedding_model = load_embedding_model()

# --- Two-Column Layout ---
col1, col2 = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: INPUTS (This section was missing and is now restored) ---
with col1:
    st.header("Controls")
    
    uploaded_file = st.file_uploader(
        "Upload a document", type=["pdf", "docx"])

    query = st.text_input("Enter your question",
                          placeholder="e.g., What are the key conclusions?")
    
    submit_button = st.button("Generate Response")

# --- RIGHT COLUMN: OUTPUTS (This section now correctly comes AFTER the left column) ---
with col2:
    st.header("Result")

    if submit_button:
        # Corrected and simplified logic for file upload only
        if not GROQ_API_KEY :
            st.error("⚠️ Please add your Groq API key. See instructions in the code.")
        elif not uploaded_file:
            st.error("⚠️ Please upload a document.")
        elif not query:
            st.error("⚠️ Please enter a question.")
        else:
            # All checks passed, so we can process the query
            with st.spinner("🤖 Analyzing document and generating answer..."):
                st.session_state.result = process_query(
                    GROQ_API_KEY,
                    uploaded_file,
                    query,
                    embedding_model
                )

    if st.session_state.result:
        result = st.session_state.result
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # --- STRUCTURED DISPLAY LOGIC ---
            st.success("✅ Analysis Complete!")

            st.markdown("#### Explanation")
            st.write(result.get('explanation', 'No explanation was provided.'))

            with st.expander("View Source Clause & Page Number"):
                st.markdown("##### Relevant Clause")
                st.markdown(f"> {result.get('relevant_clause', 'No relevant clause was found.')}")

                st.markdown("##### Page Number")
                page_num = result.get('page_number', 'N/A')
                st.info(f"**Found on Page:** {page_num}")
    else:
        st.info("The result will appear here once you submit a query.")
