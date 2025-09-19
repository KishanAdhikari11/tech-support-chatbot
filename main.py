import os
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama
from utils.embeddings import load_embeddings




@st.cache_resource
def init_embedder(model_name="all-MiniLM-L6-v2"):
    """Initialize embedding model with caching."""
    return SentenceTransformer(model_name)

@st.cache_resource
def init_chroma(persist_dir="./chroma_db", collection_name="helpdesk_docs"):
    """Initialize ChromaDB client and collection with caching."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection

@st.cache_resource
def init_llm(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0):
    """Initialize Llama model with caching and error handling."""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model path does not exist: {model_path}")
            st.info("Please provide a valid model path in the sidebar or download the model file.")
            return None
        return Llama(
            model_path=model_path, 
            n_ctx=n_ctx, 
            n_threads=n_threads, 
            n_gpu_layers=n_gpu_layers,
            verbose=False  # Reduce verbose output
        )
    except Exception as e:
        st.error(f"Failed to load Llama model: {str(e)}")
        st.info("Please provide a valid model path in the sidebar or download the model file.")
        return None

# -----------------------------
# 2️⃣ Load and Chunk Documents
# -----------------------------
@st.cache_data
def load_and_chunk_docs(file_path="docs.txt", chunk_size=500):
    """Load and chunk documents with caching."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Better chunking strategy
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [chunk for chunk in chunks if len(chunk) > 20]  # Filter tiny chunks
    except FileNotFoundError:
        st.error(f"Document file not found: {file_path}")
        return []

# -----------------------------
# 3️⃣ Add Documents to ChromaDB
# -----------------------------
def add_to_chroma(client, collection, chunks, embeddings):
    """Add documents to ChromaDB collection."""
    if not chunks or embeddings is None or len(embeddings) == 0:
        st.warning("No valid documents or embeddings to add to ChromaDB.")
        return
    
    existing = collection.get(include=[])  # fetch IDs
    existing_ids = set(existing["ids"]) if "ids" in existing else set()

    new_docs, new_embeddings, new_ids = [], [], []

    for i, chunk in enumerate(chunks):
        doc_id = f"doc_{i}"
        if doc_id not in existing_ids:
            new_docs.append(chunk)
            new_embeddings.append(embeddings[i].tolist())
            new_ids.append(doc_id)

    if new_docs:
        collection.add(
            documents=new_docs,
            embeddings=new_embeddings,
            ids=new_ids
        )
        # ChromaDB PersistentClient auto-persists - no need to call persist()
        st.success(f"Added {len(new_docs)} new documents to ChromaDB.")
    else:
        st.info("No new documents to add.")

# -----------------------------
# 4️⃣ Retrieve Relevant Context
# -----------------------------
def retrieve_context(collection, query, embedder, n_results=3):
    """Retrieve relevant context from ChromaDB."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding, 
        n_results=n_results,
        include=['documents', 'distances']  # Only get what we need
    )
    retrieved_docs = results.get("documents", [[]])[0]
    if not retrieved_docs:
        return ["No relevant documents found."]
    return retrieved_docs[:n_results]

# -----------------------------
# 5️⃣ Generate Answer with LLM (with Streaming)
# -----------------------------
def answer_question(llm, question, context, history, temperature=0.3, max_tokens=150):
    """Generate answer using LLM with streaming."""
    if llm is None:
        return [{"choices": [{"text": "LLM model not loaded. Please check the model path."}]}], ""
    
    # Build history string (limit to last 3 exchanges)
    history_str = ""
    for msg in history[-3:]:
        if msg['role'] == 'user':
            history_str += f"User: {msg['content']}\n"
        elif msg['role'] == 'assistant':
            history_str += f"Assistant: {msg['content']}\n"

    prompt_text = f"""You are a helpful IT support assistant. Use the following context and conversation history to answer the question concisely. Answer casual and friendly, answer casual hi, hello bye  greetings in respectful way.
    Also dont anything if the question is not related to IT support and not in context.

Context:
{context}

History:
{history_str}

Question:
{question}

Answer:"""

    # Stream the response
    response_stream = llm.create_completion(
        prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n"],
        stream=True
    )
    return response_stream, prompt_text

# -----------------------------
# 6️⃣ Main Streamlit App
# -----------------------------
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="IT Support RAG Assistant", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    st.title("🛠️ IT Support RAG Assistant")
    st.markdown("Ask any IT question and get context-aware solutions powered by RAG!")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input(
            "Llama Model Path", 
            value="/home/you/rag_chatbot/models/llama3/llama-3.2-3b.gguf"
        )
        n_results = st.slider("Number of Retrieved Docs", min_value=1, max_value=5, value=3)
        temperature = st.slider("LLM Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=150, step=25)
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.info("💡 For faster inference, use GPU layers (n_gpu_layers) and smaller models.")

    try:
        embedder = init_embedder()
        client, collection = init_chroma()
        llm = init_llm(model_path) if model_path and os.path.exists(model_path) else None
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return

    # Load documents once per session
    if "docs_loaded" not in st.session_state:
        with st.spinner("Loading and processing documents..."):
            chunks = load_and_chunk_docs()
            if chunks:
                embeddings = load_embeddings(embedder, chunks)
                if embeddings is not None:
                    add_to_chroma(client, collection, chunks, embeddings)
            st.session_state.docs_loaded = True

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input via chat
    if user_question := st.chat_input("Enter your IT question:"):
        if llm is None:
            with st.chat_message("assistant"):
                st.error("🚨 LLM model not loaded. Please provide a valid model path in the sidebar.")
        else:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Retrieve context
            with st.spinner("Searching knowledge base..."):
                retrieved_docs = retrieve_context(collection, user_question, embedder, n_results=n_results)
                context = "\n".join(retrieved_docs)

   
            # Generate and stream answer
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    response_stream, prompt_text = answer_question(
                        llm, user_question, context, 
                        st.session_state.messages, 
                        temperature=temperature, 
                        max_tokens=max_tokens
                    )
                    
                    for response_chunk in response_stream:
                        chunk_text = response_chunk["choices"][0]["text"]
                        full_response += chunk_text
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response.strip())
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg

            st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})


# Run the app
if __name__ == "__main__":
    main()