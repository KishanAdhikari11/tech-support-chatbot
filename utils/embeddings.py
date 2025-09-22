import os
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

def load_embeddings(embedder, chunks, embeddings_file="embeddings.pkl"):
    """
    Load or generate embeddings for document chunks.
    
    Args:
        embedder: SentenceTransformer model instance
        chunks: List of text chunks to embed
        embeddings_file: Path to save/load embeddings pickle file
    
    Returns:
        numpy.ndarray: Array of embeddings
    """
    try:
        # Check if embeddings file exists and is valid
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    saved_data = pickle.load(f)
                
                # Validate saved data structure
                if isinstance(saved_data, dict) and 'embeddings' in saved_data and 'chunk_count' in saved_data:
                    if saved_data['chunk_count'] == len(chunks):
                        st.info(f"Loaded existing embeddings for {len(chunks)} chunks.")
                        return np.array(saved_data['embeddings'])
                    else:
                        st.warning(f"Chunk count mismatch. Regenerating embeddings.")
                elif isinstance(saved_data, (list, np.ndarray)):
                    # Legacy format - just embeddings array
                    embeddings_array = np.array(saved_data)
                    if len(embeddings_array) == len(chunks):
                        st.info(f"Loaded existing embeddings for {len(chunks)} chunks.")
                        return embeddings_array
                    else:
                        st.warning(f"Chunk count mismatch. Regenerating embeddings.")
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                st.warning(f"Could not load existing embeddings: {str(e)}. Regenerating.")
        
        # Generate new embeddings
        if not chunks:
            st.error("No chunks provided for embedding generation.")
            return None
        
        st.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        progress_bar = st.progress(0)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = embedder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            
            # Update progress
            progress = min(1.0, (i + batch_size) / len(chunks))
            progress_bar.progress(progress)
        
        progress_bar.empty()
        
        embeddings_array = np.array(all_embeddings)
        
        # Save embeddings with metadata
        try:
            save_data = {
                'embeddings': embeddings_array.tolist(),
                'chunk_count': len(chunks),
                'embedding_dim': embeddings_array.shape[1],
                'model_name': embedder.get_sentence_embedding_dimension()
            }
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            st.success(f"Generated and saved embeddings for {len(chunks)} chunks.")
        except Exception as e:
            st.warning(f"Could not save embeddings: {str(e)}. Continuing without saving.")
        
        return embeddings_array
    
    except Exception as e:
        st.error(f"Error in load_embeddings: {str(e)}")
        return None

def save_embeddings(embeddings, chunks, embeddings_file="embeddings.pkl"):
    """
    Save embeddings to pickle file with metadata.
    
    Args:
        embeddings: numpy array of embeddings
        chunks: List of text chunks
        embeddings_file: Path to save embeddings
    """
    try:
        save_data = {
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'chunk_count': len(chunks),
            'embedding_dim': embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings[0])
        }
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        return True
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        return False

def validate_embeddings(embeddings, chunks):
    """
    Validate that embeddings match the expected format and chunk count.
    
    Args:
        embeddings: Embedding array to validate
        chunks: List of text chunks
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if embeddings is None:
            return False
        
        embeddings_array = np.array(embeddings)
        
        # Check dimensions
        if len(embeddings_array.shape) != 2:
            return False
        
        # Check count matches chunks
        if len(embeddings_array) != len(chunks):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings_array)) or np.any(np.isinf(embeddings_array)):
            return False
        
        return True
    
    except Exception:
        return False