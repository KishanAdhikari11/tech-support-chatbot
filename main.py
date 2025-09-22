import os
import yaml
import streamlit as st
import nltk
import tiktoken
import time
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama
from datetime import date, datetime
from utils.embeddings import load_embeddings

# Initialize logging for KPI tracking
logging.basicConfig(filename="rag_kpi.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize NLTK and tiktoken for chunking
nltk.download('punkt', quiet=True)
tokenizer = tiktoken.get_encoding("cl100k_base")

@st.cache_resource
def init_embedder(model_path="models/all-MiniLM-L6-v2", model_name="all-MiniLM-L6-v2"):
    """Initialize embedding model, trying local path first, then downloading."""
    try:
        logging.info(f"Attempting to load SentenceTransformer from: {model_path}")
        if os.path.exists(model_path):
            embedder = SentenceTransformer(model_path)
            logging.info("SentenceTransformer loaded from local path")
        else:
            logging.warning(f"Local model path {model_path} not found, downloading {model_name}")
            embedder = SentenceTransformer(model_name)
            logging.info("SentenceTransformer downloaded and loaded")
        return embedder
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        logging.error(f"Error loading SentenceTransformer: {str(e)}")
        return None

@st.cache_resource
def init_chroma(persist_dir="chroma_db", collection_name="helpdesk_docs"):
    """Initialize ChromaDB client and collection with caching."""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(name=collection_name)
        logging.info(f"Initialized ChromaDB collection: {collection_name}")
        return client, collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        logging.error(f"Error initializing ChromaDB: {str(e)}")
        return None, None

@st.cache_resource
def init_llm(model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0):
    """Initialize Llama model with caching and error handling."""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model path does not exist: {model_path}")
            logging.error(f"Model path does not exist: {model_path}")
            return None
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        logging.info("Llama model loaded successfully")
        return llm
    except Exception as e:
        st.error(f"Failed to load Llama model: {str(e)}")
        logging.error(f"Failed to load Llama model: {str(e)}")
        return None

@st.cache_data
def load_and_chunk_docs(corpus_dir="corpus/", chunk_size=750, overlap=75):
    """Load and chunk Markdown files from corpus directory."""
    try:
        if not os.path.exists(corpus_dir):
            st.error(f"Corpus directory {corpus_dir} does not exist.")
            logging.error(f"Corpus directory {corpus_dir} does not exist.")
            return [], []
        
        valid_files = [f for f in os.listdir(corpus_dir) if f.endswith((".md", ".txt", ".markdown"))]
        if not valid_files:
            st.error(f"No valid files (.md, .txt, .markdown) found in {corpus_dir}.")
            logging.error(f"No valid files found in {corpus_dir}. Contents: {os.listdir(corpus_dir)}")
            return [], []
        
        logging.info(f"Found {len(valid_files)} valid files: {valid_files}")
        chunks = []
        metadatas = []
        
        for filename in valid_files:
            file_path = os.path.join(corpus_dir, filename)
            logging.info(f"Processing file: {filename}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                if not text.strip():
                    logging.warning(f"File {filename} is empty or contains only whitespace.")
                    continue
                
                # Extract metadata
                metadata = {"source": filename, "type": "detail"}
                if text.startswith("metadata:") or text.startswith("---"):
                    try:
                        if text.startswith("metadata:"):
                            yaml_end = text.find("\n---")
                            yaml_content = text[9:yaml_end].strip() if yaml_end != -1 else text[9:].strip()
                            content = text[yaml_end + 4:].strip() if yaml_end != -1 else ""
                        else:
                            parts = text.split("---", 2)
                            yaml_content = parts[1].strip() if len(parts) >= 3 else ""
                            content = parts[2].strip() if len(parts) >= 3 else text
                        
                        if yaml_content:
                            parsed_metadata = yaml.safe_load(yaml_content)
                            if isinstance(parsed_metadata, dict):
                                # Sanitize metadata values
                                sanitized_metadata = {}
                                for key, value in parsed_metadata.items():
                                    if isinstance(value, (date, datetime)):
                                        sanitized_metadata[key] = value.isoformat()
                                    elif isinstance(value, (str, int, float, bool)) or value is None:
                                        sanitized_metadata[key] = value
                                    else:
                                        logging.warning(f"Skipping metadata key {key} in {filename}: unsupported type {type(value)}")
                                metadata.update(sanitized_metadata)
                    except yaml.YAMLError as e:
                        logging.warning(f"Invalid YAML in {filename}: {str(e)}")
                        content = text
                else:
                    content = text
                
                if not content.strip():
                    logging.warning(f"No content after parsing {filename}.")
                    continue
                
                # Token-based chunking
                sentences = nltk.sent_tokenize(content)
                current_chunk = []
                current_tokens = []
                
                for sentence in sentences:
                    sentence_tokens = tokenizer.encode(sentence)
                    if len(current_tokens) + len(sentence_tokens) <= chunk_size:
                        current_chunk.append(sentence)
                        current_tokens.extend(sentence_tokens)
                    else:
                        if current_chunk:
                            chunk_text = " ".join(current_chunk).strip()
                            chunk_tokens = len(tokenizer.encode(chunk_text))
                            logging.info(f"Chunk from {filename}: {chunk_tokens} tokens")
                            if chunk_tokens > 20:
                                chunks.append(chunk_text)
                                metadatas.append(metadata.copy())
                            else:
                                logging.warning(f"Chunk from {filename} skipped: {chunk_tokens} tokens (too small)")
                            
                            overlap_tokens = current_tokens[-overlap:] if len(current_tokens) >= overlap else current_tokens
                            if overlap_tokens:
                                overlap_text = tokenizer.decode(overlap_tokens)
                                current_chunk = [overlap_text, sentence]
                                current_tokens = overlap_tokens + sentence_tokens
                            else:
                                current_chunk = [sentence]
                                current_tokens = sentence_tokens
                
                # Add final chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    chunk_tokens = len(tokenizer.encode(chunk_text))
                    logging.info(f"Final chunk from {filename}: {chunk_tokens} tokens")
                    if chunk_tokens > 20:
                        chunks.append(chunk_text)
                        metadatas.append(metadata.copy())
                    else:
                        logging.warning(f"Final chunk from {filename} skipped: {chunk_tokens} tokens (too small)")
                
            except Exception as e:
                logging.warning(f"Error processing file {filename}: {str(e)}")
                continue
        
        if chunks:
            st.success(f"Loaded {len(chunks)} chunks from {len(set(meta['source'] for meta in metadatas))} documents.")
            logging.info(f"Loaded {len(chunks)} chunks from corpus.")
        else:
            st.error("No valid chunks created from documents.")
            logging.error("No valid chunks created from documents.")
        return chunks, metadatas
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        logging.error(f"Error loading documents: {str(e)}")
        return [], []

def add_to_chroma(client, collection, chunks, embeddings, metadatas=None):
    """Add documents to ChromaDB collection with metadata."""
    if not chunks:
        logging.error("No chunks provided to add_to_chroma")
        st.warning("No chunks provided to ChromaDB.")
        return
    if embeddings is None or len(embeddings) == 0:
        logging.error("No valid embeddings provided to add_to_chroma")
        st.warning("No valid embeddings to add to ChromaDB.")
        return
    logging.info(f"Attempting to add {len(chunks)} chunks to ChromaDB")
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing.get("ids", []))
        new_docs, new_embeddings, new_ids, new_metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            doc_id = f"doc_{i}"
            if doc_id not in existing_ids:
                new_docs.append(chunk)
                new_embeddings.append(embeddings[i].tolist())
                new_ids.append(doc_id)
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {"source": "unknown", "type": "detail"}
                # Sanitize metadata values
                sanitized_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        sanitized_metadata[key] = value
                    else:
                        sanitized_metadata[key] = str(value)
                new_metadatas.append(sanitized_metadata)
        if new_docs:
            collection.add(
                documents=new_docs,
                embeddings=new_embeddings,
                ids=new_ids,
                metadatas=new_metadatas
            )
            st.success(f"Added {len(new_docs)} new documents to ChromaDB.")
            logging.info(f"Added {len(new_docs)} chunks to ChromaDB.")
        else:
            st.info("No new documents to add.")
    except Exception as e:
        logging.error(f"Error adding to ChromaDB: {str(e)}")
        st.error(f"Error adding to ChromaDB: {str(e)}")

def retrieve_context(collection, query, embedder, n_results=3, history=None):
    """Retrieve relevant context from ChromaDB."""
    try:
        if embedder is None:
            logging.error("Embedder is None, cannot retrieve context")
            return ["Error: Embedding model not initialized."]
        
        # Check collection contents
        collection_info = collection.get(include=["documents", "metadatas"])
        logging.info(f"Collection contains {len(collection_info.get('ids', []))} documents")
        if not collection_info.get("ids", []):
            logging.error("ChromaDB collection is empty")
            return ["No documents found in ChromaDB collection."]
        
        # Combine query with recent history
        query_text = query
        if history:
            recent_history = " ".join([msg["content"] for msg in history[-2:] if msg["role"] == "user"])
            if recent_history:
                query_text = f"{query} {recent_history}"
        
        query_embedding = embedder.encode([query_text]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'distances', 'metadatas']
        )
        
        if results is None:
            logging.error("ChromaDB query returned None")
            return ["No documents found in ChromaDB due to query failure."]
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            logging.info(f"No documents retrieved for query: {query}")
            return ["No relevant documents found in the knowledge base."]
        
        sources = [meta.get("source", "unknown") for meta in metadatas]
        logging.info(f"Query: {query}, Retrieved sources: {sources}")
        return documents
    
    except Exception as e:
        logging.error(f"Error retrieving context: {str(e)}")
        return [f"Error retrieving relevant information: {str(e)}"]

def answer_question(llm, question, context, history, temperature=0.3, max_tokens=150):
    """Generate answer using LLM."""
    if llm is None:
        return "LLM model not loaded. Please check the model path.", ""
    
    try:
        # Build history string (limit to last 3 exchanges)
        history_str = ""
        for msg in history[-3:]:
            if msg['role'] == 'user':
                history_str += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                history_str += f"Assistant: {msg['content']}\n"

        # Create context string
        context_str = "\n".join(context[:3])
        print(context_str)

        prompt_text = f"""You are a helpful IT support assistant. Use the provided context and conversation history to answer IT-related questions with clear, step-by-step solutions. If the question is not IT-related, politely redirect the user.

Context:
{context_str}

Previous Conversation:
{history_str}

Current Question: {question}

Answer (be concise and helpful):"""

        # Generate response
        start_time = time.time()
        response = llm.create_completion(
            prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\nUser:", "\n\nQuestion:", "###"]
        )
        response_time = time.time() - start_time
        
        # Extract response text
        if isinstance(response, dict) and "choices" in response:
            answer = response["choices"][0]["text"].strip()
        else:
            answer = str(response).strip()
        
        logging.info(f"Response time for query '{question}': {response_time:.2f}s")
        return answer, prompt_text
    
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logging.error(error_msg)
        return error_msg, ""

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="IT Support RAG Assistant", layout="wide", initial_sidebar_state="expanded")
    st.title("🛠️ IT Support RAG Assistant")
    st.markdown("Ask any IT question and get context-aware solutions powered by RAG!")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Llama Model Path", value="models/llama3/llama-3.2-3b.gguf")
        embedder_path = st.text_input("Embedding Model Path", value="models/all-MiniLM-L6-v2")
        n_results = st.slider("Number of Retrieved Docs", min_value=1, max_value=5, value=3)
        temperature = st.slider("LLM Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=150, step=25)
        
        if st.button("Clear Model and Document Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.docs_loaded = False
            st.rerun()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.info("💡 For faster inference, use GPU layers or smaller models.")
        
        # Show corpus info
        if os.path.exists("corpus"):
            corpus_files = [f for f in os.listdir("corpus") if f.endswith(('.md', '.txt', '.markdown'))]
            st.info(f"📚 Corpus: {len(corpus_files)} documents loaded")

    # Initialize models
    try:
        embedder = init_embedder(model_path=embedder_path)
        if embedder is None:
            st.error("Failed to initialize embedder. Check model path or network.")
            return
        client, collection = init_chroma()
        if client is None or collection is None:
            st.error("Failed to initialize ChromaDB.")
            return
        llm = init_llm(model_path) if model_path and os.path.exists(model_path) else None
        if llm is None and model_path:
            st.warning("⚠️ LLM model not loaded. Please check the model path.")
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return

    # Load documents once per session
    if "docs_loaded" not in st.session_state:
        with st.spinner("Loading and processing documents..."):
            chunks, metadatas = load_and_chunk_docs()
            if chunks:
                embeddings = load_embeddings(embedder, chunks)
                if embeddings is not None:
                    add_to_chroma(client, collection, chunks, embeddings, metadatas)
                else:
                    st.error("Failed to generate embeddings.")
            else:
                st.warning("No documents loaded from corpus.")
            st.session_state.docs_loaded = True

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_question := st.chat_input("Enter your IT question:"):
        if llm is None:
            with st.chat_message("assistant"):
                st.error("🚨 LLM model not loaded. Please provide a valid model path.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("Searching knowledge base..."):
                retrieved_docs = retrieve_context(collection, user_question, embedder, n_results=n_results, history=st.session_state.messages)

            with st.chat_message("assistant"):
                try:
                    full_response, prompt_used = answer_question(
                        llm, user_question, retrieved_docs, st.session_state.messages, temperature=temperature, max_tokens=max_tokens
                    )
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()