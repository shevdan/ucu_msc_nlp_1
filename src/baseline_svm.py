import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

TRAIN_PATH = "emotions/train.csv"
TEST_PATH  = "emotions/test.csv"
OUT_PATH   = "submissions/submission_tfidf.csv"

def fit_and_eval(texts, labels, seed=42):
    X_tr, X_va, y_tr, y_va = train_test_split(
        texts, labels, test_size=0.15, random_state=seed, stratify=labels
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
        )),
        ("svm", LinearSVC())
    ])

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)

    f1 = f1_score(y_va, pred, average="macro")
    print("Macro-F1:", f1)
    print(classification_report(y_va, pred))
    return clf, f1

def main():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    text = train["text"].astype(str).tolist()

    emotion_model, _ = fit_and_eval(text, train["emotion"].astype(str).tolist())

    category_model, _ = fit_and_eval(text, train["category"].astype(str).tolist())

    test_text = test["text"].astype(str).tolist()
    sub = pd.DataFrame({
        "index": test["Unnamed: 0"],
        "emotion": emotion_model.predict(test_text),
        "category": category_model.predict(test_text),
    })

    sub.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
