import fitz
import numpy as np
import faiss
import torch
import streamlit as st
import random
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'ibm-granite/granite-3.0-2b-instruct'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# --- MODEL LOADING (with Caching and Quantization) ---
@st.cache_resource
def load_models():
    """Load and cache the embedding and quantized LLM models."""
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading LLM and tokenizer for {LLM_MODEL_NAME}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Models loaded successfully!")
    return embedding_model, tokenizer, llm_model

# --- BACKEND LOGIC ---
def extract_text_from_uploaded_pdfs(uploaded_files):
    all_texts = {}
    for uploaded_file in uploaded_files:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            all_texts[uploaded_file.name] = text
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
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

def retrieve_relevant_chunks(query, index, embedding_model, all_chunks, top_k):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [all_chunks[i] for i in indices[0]]

def construct_prompt_for_qna(query, context_chunks):
    context = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" for chunk in context_chunks])
    prompt = f"""
<|user|>
Use the following context to answer the question. If the answer is not in the context, state that the answer is not found in the provided documents.
Context:
{context}
Question:
{query}
<|assistant|>
"""
    return prompt

def generate_answer(prompt, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text[len(prompt):].strip()

# NEW: Function to check if the content is academic
def is_study_material(text_sample, llm_model, tokenizer):
    """Uses the LLM to quickly classify if the text is academic/study material."""
    prompt = f"""
<|user|>
Based on the following text sample, is this academic, technical, or study-related material? Answer with only the word "yes" or "no".

Text sample:
"{text_sample[:1000]}"
<|assistant|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=3) # Only need a short answer
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip().lower()
    return "yes" in answer

def generate_quiz_questions(context_chunk, llm_model, tokenizer):
    """Generates a quiz based on a single text chunk."""
    context = context_chunk['text']
    prompt = f"""
<|user|>
You are an expert quiz creator. Based on the following context, generate a JSON object containing a list of 3 multiple-choice questions.
Each question should have a 'question' field, an 'options' field with 4 possible answers, and a 'correct_answer' field with the letter of the correct option (e.g., 'A', 'B', 'C', 'D').
The questions must be answerable only using the provided context.

Context:
{context}
<|assistant|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=768, temperature=0.5)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response_text[len(prompt):].strip()

    try:
        json_start = response_only.find('{')
        json_end = response_only.rfind('}') + 1
        json_str = response_only[json_start:json_end]
        quiz_data = json.loads(json_str)
        return quiz_data.get("questions", [])
    except (json.JSONDecodeError, IndexError):
        st.error("Failed to generate a valid quiz from the document content.")
        return None

# --- STREAMLIT UI ---
st.set_page_config(page_title="StudyMate by Napstablook Syndicate", layout="wide")
st.title("📚 StudyMate: AI-Powered PDF Q&A")

embedding_model, tokenizer, llm_model = load_models()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
# NEW: State to track content type
if "is_academic" not in st.session_state:
    st.session_state.is_academic = False

with st.sidebar:
    st.header("1. Upload Your PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents... This may take a moment."):
            raw_texts = extract_text_from_uploaded_pdfs(uploaded_files)
            all_chunks = chunk_texts(raw_texts, CHUNK_SIZE, CHUNK_OVERLAP)
            if all_chunks:
                faiss_index = build_faiss_index(all_chunks, embedding_model)
                st.session_state.processed_data = {"faiss_index": faiss_index, "all_chunks": all_chunks}
                
                # NEW: Check if the material is academic after processing
                with st.spinner("Analyzing content type..."):
                    st.session_state.is_academic = is_study_material(all_chunks[0]['text'], llm_model, tokenizer)

                st.session_state.messages = []
                st.session_state.quiz_questions = None
                st.session_state.quiz_submitted = False
                st.success("✅ Documents processed successfully!")
            else:
                st.warning("Could not extract any text from the uploaded PDF(s).")
                st.session_state.processed_data = None


if st.session_state.processed_data:
    tab1, tab2 = st.tabs(["Q&A Chat", "Quiz Session"])

    with tab1:
        # Q&A chat logic remains the same
        st.header("💬 Ask Questions About Your Documents")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.info(f"**Source:** {source['source']}\n\n**Content:** {source['text']}")
        
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    faiss_index = st.session_state.processed_data["faiss_index"]
                    all_chunks = st.session_state.processed_data["all_chunks"]
                    relevant_chunks = retrieve_relevant_chunks(prompt, faiss_index, embedding_model, all_chunks, TOP_K)
                    full_prompt = construct_prompt_for_qna(prompt, relevant_chunks)
                    answer = generate_answer(full_prompt, llm_model, tokenizer)
                    
                    st.markdown(answer)
                    with st.expander("View Sources"):
                        for chunk in relevant_chunks:
                            st.info(f"**Source:** {chunk['source']}\n\n**Content:** {chunk['text']}")

                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": relevant_chunks})

    with tab2:
        st.header("🧠 Test Your Knowledge!")
        # MODIFIED: Conditionally show quiz generation based on content type
        if st.session_state.is_academic:
            if st.button("Generate New Quiz"):
                with st.spinner("Creating a quiz from your documents..."):
                    all_chunks = st.session_state.processed_data["all_chunks"]
                    # MODIFIED: Use only one random chunk for much faster generation
                    quiz_context_chunk = random.sample(all_chunks, 1)[0]
                    st.session_state.quiz_questions = generate_quiz_questions(quiz_context_chunk, llm_model, tokenizer)
                    st.session_state.quiz_submitted = False
            
            if st.session_state.quiz_questions:
                if st.session_state.quiz_submitted:
                    st.subheader("Quiz Results:")
                    score = 0
                    for i, q in enumerate(st.session_state.quiz_questions):
                        user_ans = st.session_state.user_answers.get(f"question_{i}")
                        correct_ans_text = next((opt for opt in q["options"] if opt.startswith(q["correct_answer"])), "N/A")
                        
                        if user_ans and user_ans.startswith(q["correct_answer"]):
                            score += 1
                            st.success(f"**Question {i+1}:** {q['question']} - Correct!")
                            st.write(f"Your answer: {user_ans}")
                        else:
                            st.error(f"**Question {i+1}:** {q['question']} - Incorrect.")
                            st.write(f"Your answer: {user_ans if user_ans else 'No answer'}")
                            st.write(f"Correct answer: {correct_ans_text}")
                    st.metric(label="Your Score", value=f"{score}/{len(st.session_state.quiz_questions)}")
                else:
                    with st.form("quiz_form"):
                        st.subheader("Answer the questions below:")
                        user_answers = {}
                        for i, q in enumerate(st.session_state.quiz_questions):
                            st.markdown(f"**{i+1}. {q['question']}**")
                            options = [f"{opt}" for opt in q["options"]]
                            user_answers[f"question_{i}"] = st.radio("Options:", options, key=f"q_{i}", label_visibility="collapsed")

                        submitted = st.form_submit_button("Submit Quiz")
                        if submitted:
                            st.session_state.user_answers = user_answers
                            st.session_state.quiz_submitted = True
                            st.rerun()
        else:
            # NEW: Message for non-study material
            st.warning("Cannot generate a quiz. The uploaded document does not appear to be study material.")

else:
    st.info("👋 Welcome to StudyMate! Please upload and process your PDF documents in the sidebar to begin.")