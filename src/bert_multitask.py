import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "youscan/ukr-roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
SEED = 42

TRAIN_PATH = "emotions/train.csv"
TEST_PATH = "emotions/test.csv"
OUT_PATH = "submissions/submission_bert.csv"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, emo2id, cat2id, train=True):
        self.texts = df["text"].astype(str).tolist()
        self.train = train
        self.tokenizer = tokenizer

        if train:
            self.emotions = df["emotion"].map(emo2id).tolist()
            self.categories = df["category"].map(cat2id).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.train:
            item["labels_emotion"] = torch.tensor(self.emotions[idx])
            item["labels_category"] = torch.tensor(self.categories[idx])

        return item

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name, n_emotions, n_categories):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.emotion_head = nn.Linear(hidden, n_emotions)
        self.category_head = nn.Linear(hidden, n_categories)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_emotion=None,
        labels_category=None,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)

        logits_emotion = self.emotion_head(cls)
        logits_category = self.category_head(cls)

        loss = None
        if labels_emotion is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = (
                loss_fct(logits_emotion, labels_emotion)
                + loss_fct(logits_category, labels_category)
            )

        return {
            "loss": loss,
            "logits_emotion": logits_emotion,
            "logits_category": logits_category,
        }

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            loss = outputs.get("loss")
            logits_emotion = outputs["logits_emotion"]
            logits_category = outputs["logits_category"]

        labels_emotion = inputs.get("labels_emotion")
        labels_category = inputs.get("labels_category")

        if labels_emotion is not None:
            labels = (labels_emotion, labels_category)
        else:
            labels = None

        return (loss, (logits_emotion, logits_category), labels)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

emotions = sorted(train_df["emotion"].unique())
categories = sorted(train_df["category"].unique())

emo2id = {e: i for i, e in enumerate(emotions)}
cat2id = {c: i for i, c in enumerate(categories)}
id2emo = {i: e for e, i in emo2id.items()}
id2cat = {i: c for c, i in cat2id.items()}

train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df["emotion"],
    random_state=SEED,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = ReviewDataset(train_df, tokenizer, emo2id, cat2id, train=True)
val_ds = ReviewDataset(val_df, tokenizer, emo2id, cat2id, train=True)

model = MultiTaskBERT(
    MODEL_NAME,
    n_emotions=len(emotions),
    n_categories=len(categories),
)

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = MultiTaskTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

pred = trainer.predict(val_ds)
logits_emotion, logits_category = pred.predictions
labels_emotion, labels_category = pred.label_ids

pred_e = logits_emotion.argmax(1)
pred_c = logits_category.argmax(1)

f1_e = f1_score(labels_emotion, pred_e, average="macro")
f1_c = f1_score(labels_category, pred_c, average="macro")

print("Emotion F1:", f1_e)
print("Category F1:", f1_c)
print("Mean F1:", (f1_e + f1_c) / 2)

test_ds = ReviewDataset(test_df, tokenizer, emo2id, cat2id, train=False)
test_pred = trainer.predict(test_ds)
logits_emotion_test, logits_category_test = test_pred.predictions

pred_emotion = logits_emotion_test.argmax(1)
pred_category = logits_category_test.argmax(1)

submission = pd.DataFrame({
    "index": range(len(test_df)),
    "emotion": [id2emo[i] for i in pred_emotion],
    "category": [id2cat[i] for i in pred_category],
})

submission.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
