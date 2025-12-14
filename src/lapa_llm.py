import json
import os
import pickle
import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import re

LAPA_MODEL = "lapa-llm/lapa-v0.1.2-instruct"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

FAISS_INDEX = "faiss.index"
FAISS_META = "faiss_meta.pkl"

TEST_PATH = "emotions/test.csv"
OUT_PATH = "submissions/submission_lapa_faiss.csv"

TOP_K = 3
MAX_PROMPT_LEN = 512
MAX_NEW_TOKENS = 48

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

CATEGORIES = ['Complaint / Dissatisfaction', 'Gratitude / Positive Feedback', 'Neutral Comment', 'Question / Request for Help', 'Suggestion / Idea']

SYSTEM_PROMPT = (
    "You are a strict classifier for Ukrainian short reviews.\n"
    "You must output ONLY valid JSON.\n"
    "The JSON must have exactly two fields: 'emotion' and 'category'.\n"
    "Do NOT repeat the input text"
    "Use ONLY the labels provided.\n"
    "Do NOT explain.\n\n"
)

def truncate(text, max_chars=300):
    return text[:max_chars].replace("\n", " ")

def build_prompt(query, examples):
    block = ""
    for i, ex in enumerate(examples, 1):
        block += (
            f"Example {i}:\n"
            f"Text: {truncate(ex['text'])}\n"
            f"Emotion: {ex['emotion']}\n"
            f"Category: {ex['category']}\n\n"
        )

    return (
        SYSTEM_PROMPT +
        f"Allowed emotions: {EMOTIONS}\n"
        f"Allowed categories: {CATEGORIES}\n\n"
        f"Here are similar labeled examples:\n"
        f"{block}"
        f"Now classify this review:\n"
        f"Text: {truncate(query)}\n\n"
        f"Answer:"
    )

def extract_first_json(text):
    for i in range(len(text)):
        if text[i] == "{":
            for j in range(i + 1, len(text)):
                if text[j] == "}":
                    try:
                        return json.loads(text[i:j+1])
                    except Exception:
                        continue
    return None

def parse_lapa_output(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end+1])
            emo = obj.get("emotion")
            cat = obj.get("category")
            return emo, cat
        except Exception:
            pass

    try:
        obj = extract_first_json(text)
        if obj:
            return obj.get("emotion"), obj.get("category")
    except Exception:
        pass

    emo_match = re.search(r"Emotion\s*:\s*(.+)", text, re.IGNORECASE)
    cat_match = re.search(r"Category\s*:\s*(.+)", text, re.IGNORECASE)

    emo = emo_match.group(1).strip() if emo_match else None
    cat = cat_match.group(1).strip() if cat_match else None

    if emo or cat:
        return emo, cat

    return None, None

def load_existing_predictions(path):
    if not os.path.exists(path):
        return {}, {}

    df = pd.read_csv(path)
    emo_map = dict(zip(df["index"], df["emotion"]))
    cat_map = dict(zip(df["index"], df["category"]))
    return emo_map, cat_map

def main():
    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_META, "rb") as f:
        meta = pickle.load(f)

    train_texts = meta["texts"]
    train_emotions = meta["emotions"]
    train_categories = meta["categories"]

    emb_model = SentenceTransformer(EMB_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(LAPA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LAPA_MODEL,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    test = pd.read_csv(TEST_PATH)

    done_emotions, done_categories = load_existing_predictions(OUT_PATH)
    done_indices = set(done_emotions.keys())

    print(f"Already processed: {len(done_indices)} / {len(test)}")

    file_exists = os.path.exists(OUT_PATH)
    out_f = open(OUT_PATH, "a", newline="", encoding="utf-8")

    if not file_exists:
        out_f.write("index,emotion,category\n")
        out_f.flush()

    for idx, text in tqdm(
        enumerate(test["text"].astype(str).tolist()),
        total=len(test),
        desc="Running RAG Lapa",
    ):
        if idx in done_indices:
            continue

        q_emb = emb_model.encode(
            [text],
            normalize_embeddings=True
        ).astype("float32")

        _, idxs = index.search(q_emb, TOP_K)
        idxs = idxs[0]

        examples = [
            {
                "text": train_texts[i],
                "emotion": train_emotions[i],
                "category": train_categories[i],
            }
            for i in idxs
        ]

        prompt = build_prompt(text, examples)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        emo, cat = parse_lapa_output(decoded)

        if emo not in EMOTIONS:
            emo = "Neutral"
        if cat not in CATEGORIES:
            cat = "Neutral Comment"

        out_f.write(f"{idx},{emo},{cat}\n")
        out_f.flush()

    out_f.close()
    print("Finished. Saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
