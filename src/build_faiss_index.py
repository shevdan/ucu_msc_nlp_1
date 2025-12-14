import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

TRAIN_PATH = "emotions/train.csv"
OUT_INDEX = "faiss.index"
OUT_META = "faiss_meta.pkl"

EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    df = pd.read_csv(TRAIN_PATH)

    texts = df["text"].astype(str).tolist()
    emotions = df["emotion"].astype(str).tolist()
    categories = df["category"].astype(str).tolist()

    model = SentenceTransformer(EMB_MODEL)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.array(embeddings, dtype=np.float32, copy=True)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, OUT_INDEX)

    meta = {
        "texts": texts,
        "emotions": emotions,
        "categories": categories,
    }

    with open(OUT_META, "wb") as f:
        pickle.dump(meta, f)

    print("FAISS index saved:", OUT_INDEX)
    print("Metadata saved:", OUT_META)
    print("Num vectors:", index.ntotal)

if __name__ == "__main__":
    main()
