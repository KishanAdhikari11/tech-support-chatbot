import os
import pickle



def load_embeddings(embedder, chunks, cache_file="embeddings.pkl"):
    """Load embeddings from cache or compute if not exists."""
    if not chunks:
        return None
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=True)
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings