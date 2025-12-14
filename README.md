# Ukrainian Reviews Classification

Multi-task classification of Ukrainian reviews by **emotion** and **category** for the UCU NLP course Kaggle competition.

## Task Description

Classify short Ukrainian reviews into:
- **Emotions** (7 classes): Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
- **Categories** (5 classes): Complaint/Dissatisfaction, Gratitude/Positive Feedback, Neutral Comment, Question/Request for Help, Suggestion/Idea

**Metric**: Macro F1 averaged across both tasks.

## Dataset

| Split | Samples |
|-------|---------|
| Train | 9,335 |
| Test  | 4,064 |

**Class imbalance**: Significant imbalance exists (e.g., Happiness: ~1,600 samples vs Fear: ~9 samples).

**Text length**: Median 30 tokens, 99.5% under 86 tokens.

## Results

| Experiment | Score | Description |
|------------|-------|-------------|
| TF-IDF + SVM | 0.47 | Baseline with classic ML |
| BERT Multi-task | 0.56 | Single BERT with shared encoder, two heads |
| BERT Separate | 0.60 | Two independent BERT models |
| Lapa LLM + FAISS (English) | **0.626** | Few-shot with Ukrainian LLM LAPA, English prompts |
| Lapa LLM + FAISS (Ukrainian) | 0.55 | Few-shot with Ukrainian LLM LAPA, Ukrainian prompts |

## Hardware
I used RTX4070 12GB VRAM for my experiments. For LLM experiments it took around 2.5-3 hours to complete.

## Experiments

### 1. TF-IDF + SVM Baseline

Classic approach using TF-IDF vectorization with Linear SVM classifiers.

**Configuration:**
- TF-IDF: bigrams (1,2), min_df=2, max_features=200,000, sublinear TF
- Model: LinearSVC for each task
- Train/Val split: 85/15 stratified on emotion

**Result**: 0.47

### 2. BERT Multi-task

Single Ukrainian RoBERTa model with shared encoder and two classification heads.

**Configuration:**
- Model: `youscan/ukr-roberta-base`
- Architecture: Shared encoder -> Dropout -> Two linear heads
- Loss: Sum of CrossEntropy losses for both tasks
- Max length: 128 tokens
- Epochs: 3, LR: 2e-5, Batch size: 16

**Result**: 0.56

### 3. BERT Separate Models 

Two independent BERT models, one for each task

**Configuration:**
- Model: `youscan/ukr-roberta-base` (separate instance per task)
- Class weights: Computed using sklearn's `balanced` strategy
- Early stopping: patience=3, metric=macro_f1
- Epochs: 20 (with early stopping), LR: 3e-5
- Warmup ratio: 0.06, Weight decay: 0.01


**Result**: 0.60

### 4. Lapa LLM + FAISS Few-shot

Retrieval-augmented generation using Ukrainian instruction-tuned LLM LAPA

**Architecture:**
1. Build FAISS index from training embeddings (`src/build_faiss_index.py`)
2. For each test sample, retrieve top-K similar training examples
3. Construct few-shot prompt with retrieved examples
4. Generate classification with Lapa LLM

**Configuration:**
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- LLM: `lapa-llm/lapa-v0.1.2-instruct`
- Top-K examples: 3
- Index: FAISS IndexFlatIP (cosine similarity via inner product)

**English prompt version:**
- System prompt and labels in English
- JSON output format requested

**Result**: 0.626 (best)

### 5. Lapa LLM + FAISS Ukrainian

Same as experiment 4, but with Ukrainian prompts and labels.

**Changes:**
- System prompt in Ukrainian
- Labels translated to Ukrainian (e.g., "Happiness" -> "Щастя")
- Output parsed from Ukrainian JSON keys ("емоція", "категорія")
- Results mapped back to English for submission

**Hypothesis**: Ukrainian prompts might work better since Lapa is instruction-tuned in Ukrainian.

**Result**: 0.55
This experiment showed worse performance than the one with prompts in Englis


## Key Findings

* Separate models > Multi-task: Training independent models for each task outperformed shared encoder approach (+0.04).

* Few-shot LLM is competitive: Lapa LLM with RAG and few-shot achieved the best score without any fine-tuning.

* English prompts > Ukrainian prompts: English prompts worked better with Lapa despite it being Ukrainian-tuned
