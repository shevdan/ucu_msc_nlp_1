import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME = "youscan/ukr-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-5
SEED = 42
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01

TRAIN_PATH = "emotions/train.csv"
TEST_PATH = "emotions/test.csv"
OUT_PATH = "submissions/submission_bert_separate.csv"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

class SingleTaskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

def train_model(task_name, train_texts, train_labels, val_texts, val_labels,
                label2id, id2label, tokenizer):
    print(f"\n{'='*50}")
    print(f"Training {task_name} classifier")
    print(f"{'='*50}")

    n_classes = len(label2id)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_ds = SingleTaskDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = SingleTaskDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=f"outputs/{task_name}",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=2,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    val_pred = trainer.predict(val_ds)
    preds = val_pred.predictions.argmax(1)
    f1 = f1_score(val_labels, preds, average="macro")
    print(f"{task_name} Macro F1: {f1:.4f}")

    return trainer, f1

if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    emotions = sorted(train_df["emotion"].unique())
    categories = sorted(train_df["category"].unique())

    emo2id = {e: i for i, e in enumerate(emotions)}
    cat2id = {c: i for i, c in enumerate(categories)}
    id2emo = {i: e for e, i in emo2id.items()}
    id2cat = {i: c for c, i in cat2id.items()}

    print(f"Emotions ({len(emotions)}): {emotions}")
    print(f"Categories ({len(categories)}): {categories}")

    train_data, val_data = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df["emotion"],
        random_state=SEED,
    )

    train_texts = train_data["text"].astype(str).tolist()
    val_texts = val_data["text"].astype(str).tolist()

    train_emotions = train_data["emotion"].map(emo2id).tolist()
    val_emotions = val_data["emotion"].map(emo2id).tolist()

    train_categories = train_data["category"].map(cat2id).tolist()
    val_categories = val_data["category"].map(cat2id).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    emotion_trainer, f1_e = train_model(
        "emotion",
        train_texts, train_emotions,
        val_texts, val_emotions,
        emo2id, id2emo,
        tokenizer
    )

    category_trainer, f1_c = train_model(
        "category",
        train_texts, train_categories,
        val_texts, val_categories,
        cat2id, id2cat,
        tokenizer
    )

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Emotion F1:  {f1_e:.4f}")
    print(f"Category F1: {f1_c:.4f}")
    print(f"Mean F1:     {(f1_e + f1_c) / 2:.4f}")

    test_texts = test_df["text"].astype(str).tolist()
    test_ds = SingleTaskDataset(test_texts, None, tokenizer, MAX_LEN)

    emotion_pred = emotion_trainer.predict(test_ds)
    pred_emotions = emotion_pred.predictions.argmax(1)

    category_pred = category_trainer.predict(test_ds)
    pred_categories = category_pred.predictions.argmax(1)

    submission = pd.DataFrame({
        "index": range(len(test_df)),
        "emotion": [id2emo[i] for i in pred_emotions],
        "category": [id2cat[i] for i in pred_categories],
    })

    submission.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
