import fitz  
import numpy as np
import faiss
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # generating 384-dim embeddings.
LLM_MODEL_NAME = 'ibm-granite/granite-3.2-8b-instruct'
CHUNK_SIZE = 500 #number of words for each text chunk
CHUNK_OVERLAP = 50 # no. of words to be overlapped between consecutive chunks
TOP_K = 3 # no. of most relevant chunks to be retrived for the context.

# This function will only run once, and the loaded models will be reused across sessions.
@st.cache_resource
def load_models():
    """Load and cache the embedding and LLM models."""
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Loading LLM and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME) # handles conversation of text into integer token ID for the model.
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto" # load balances between the cpu and the gpu dynamically
    )
    print("Models loaded successfully!")
    return embedding_model, tokenizer, llm_model

def extract_text_from_uploaded_pdfs(uploaded_files):
    """Extracts text from Streamlit UploadedFile objects."""
    all_texts = {}
    for uploaded_file in uploaded_files:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            all_texts[uploaded_file.name] = text
            print(f"  - Successfully processed '{uploaded_file.name}'")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    return all_texts

def chunk_texts(texts_dict, chunk_size, chunk_overlap):
    chunks_with_metadata = []
    for filename, text in texts_dict.items():
        words = text.split()
        if not words: continue
        step = chunk_size - chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            chunks_with_metadata.append({"text": chunk, "source": filename})
    return chunks_with_metadata

def build_faiss_index(chunks, embedding_model):
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def retrieve_relevant_chunks(query, index, embedding_model, all_chunks, top_k):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [all_chunks[i] for i in indices[0]]

def construct_prompt(query, context_chunks):
    context = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" for chunk in context_chunks])
    prompt = f"""
You are an AI study assistant. Your task is to answer the user's question based *only* on the provided context from the documents.
If the answer cannot be found in the context, clearly state that. Do not use any external knowledge.

--- CONTEXT FROM DOCUMENTS ---
{context}
--- END OF CONTEXT ---

QUESTION: {query}

ANSWER:
"""
    return prompt

def generate_answer(prompt, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to(llm_model.device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7, # for creativity
        repetition_penalty=1.1,
        do_sample=True # sampling needed for temperature to work / avoid predictable creativity.
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text[len(prompt):].strip()

# --- STREAMLIT UI ---
st.set_page_config(page_title="StudyMate by Napstablook Syndicate", layout="wide")
st.title("StudyMate: AI-Powered PDF Q&A by Napstablook Syndicate")

# Load models once and cache them
embedding_model, tokenizer, llm_model = load_models()

# Initialize session state for storing data across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

# --- Sidebar for PDF Upload and Processing ---
with st.sidebar:
    st.header("1. Upload Your PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", accept_multiple_files=True, type="pdf"
    )

    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents... This may take a moment."):
            # Extract text and create chunks
            raw_texts = extract_text_from_uploaded_pdfs(uploaded_files)
            all_chunks = chunk_texts(raw_texts, CHUNK_SIZE, CHUNK_OVERLAP)

            # Build FAISS index
            faiss_index = build_faiss_index(all_chunks, embedding_model)

            # Store processed data in session state
            st.session_state.processed_data = {
                "faiss_index": faiss_index,
                "all_chunks": all_chunks
            }
            st.session_state.messages = [] # Clear previous messages
        st.success("[INFO]: Documents processed successfully! You can now ask questions.")
    
    st.info("[Note]: Processing large documents or many files may take time.")

# --- Main Chat Interface ---
st.header("2. Ask Questions About Your Documents")

if st.session_state.processed_data is None:
    st.info("Please upload and process your PDF documents in the sidebar to begin.")

# Display existing messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(f"Source: {source['source']}\n\nContent: {source['text']}")

# Chat input for user's question
if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.processed_data:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response / MAIN Pipleline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve context
                faiss_index = st.session_state.processed_data["faiss_index"]
                all_chunks = st.session_state.processed_data["all_chunks"]
                relevant_chunks = retrieve_relevant_chunks(prompt, faiss_index, embedding_model, all_chunks, TOP_K)

                # Construct prompt and generate answer
                full_prompt = construct_prompt(prompt, relevant_chunks)
                answer = generate_answer(full_prompt, llm_model, tokenizer)
                
                # Display answer and sources
                st.markdown(answer)
                with st.expander("View Sources"):
                    for chunk in relevant_chunks:
                        st.info(f"Source: {chunk['source']}\n\nContent: {chunk['text']}")

                # Add AI response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": relevant_chunks
                })
    else:
        st.warning("You must process documents before asking questions.")