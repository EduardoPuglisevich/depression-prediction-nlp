# Depression Detection in Reddit Posts — NLP Classification

> A team project classifying Reddit posts as depressive or non-depressive language, comparing a classical interpretable baseline against a fine-tuned transformer, served through an interactive web demo.

This is a fork of a teammate's repository ([`Elysian-C/TF_Data_mining_tools`](https://github.com/Elysian-C/TF_Data_mining_tools)), built by a 3-person team. My contribution was the full modeling pipeline (`02_modeling_and_training.ipynb`): both models, hyperparameter optimization, and interpretability analysis. Exploratory data analysis and the Gradio web app were built by teammates.

| Component | Owner |
|---|---|
| Exploratory Data Analysis (`01_exploratory_data_analysis.ipynb`, Sweetviz reports) | Team |
| **Modeling & training** (`02_modeling_and_training.ipynb`) | **Eduardo Puglisevich** |
| Web application (`app.py`, Gradio) | Team |

**Stack:** Python · scikit-learn · Hugging Face Transformers · PyTorch · Optuna · SHAP

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Skills Demonstrated](#skills-demonstrated)
- [Modeling Methodology & Results](#modeling-methodology--results)
- [Ethical Considerations](#ethical-considerations)

---

Depression remains one of the most prevalent and under-diagnosed mental health conditions, and social platforms like Reddit contain large volumes of spontaneous text that may carry early linguistic signals of emotional distress. This project evaluates how well both a classical, fully interpretable model and a fine-tuned transformer can classify that signal — and whether the accuracy gain from a transformer is worth the interpretability trade-off.

## Problem Statement

- Can machine learning models accurately classify Reddit posts as depressive or non-depressive from text alone?
- How does a classical, coefficient-interpretable model compare against a fine-tuned transformer on this task?
- Which specific words or tokens drive each model's predictions, and do they align with clinically-recognized depressive language patterns?

## Dataset

- **Source:** Reddit Depression Dataset (Kaggle)
- **Original size:** ~2.47M records
- **Working dataset (after cleaning/balancing):** ~78K balanced samples
- **Labels:** `1` = depressive language, `0` = non-depressive
- Two text representations were prepared: a normalized `body` field (stemmed, stopwords removed) for the classical model, and the original `body_raw` for the transformer, which handles its own tokenization.

---

## Skills Demonstrated

*(Scoped to my contribution — the modeling pipeline)*

| Area | Where it shows up |
|---|---|
| Classical NLP baseline, built for interpretability | TF-IDF (unigrams + bigrams, 20K vocabulary) + Logistic Regression, with coefficient analysis to surface the specific words driving each prediction |
| Transformer fine-tuning | DistilBERT fine-tuned end-to-end on raw Reddit text, trained and evaluated with Hugging Face `Trainer` |
| Hyperparameter optimization, two different strategies | `RandomizedSearchCV` for the classical model (log-uniform search over regularization strength `C`); Optuna Bayesian search for DistilBERT (learning rate, batch size, weight decay) |
| Model interpretability across two paradigms | Coefficient-based feature importance for Logistic Regression; SHAP token-level attribution for DistilBERT — comparing *why* each model predicts what it predicts, not just *how well* |
| Rigorous, apples-to-apples model comparison | Both models evaluated on the same stratified 80/20 split with accuracy, precision, recall, and F1 — not cherry-picking whichever metric favors one model |
| Production model export | Both models serialized for deployment (`joblib`/`pickle` for the classical pipeline, Hugging Face's native save format for the fine-tuned transformer) |

---

## Modeling Methodology & Results

### Logistic Regression + TF-IDF
A TF-IDF vectorizer (unigrams and bigrams, 20,000-term vocabulary, min/max document frequency filtering) feeds a Logistic Regression classifier. After baseline training, `RandomizedSearchCV` tuned the regularization strength (`C`) over a log-uniform search space, optimized for F1.

### DistilBERT (fine-tuned)
DistilBERT was fine-tuned end-to-end on raw post text, then re-tuned with an Optuna study searching learning rate, batch size, and weight decay — a proper Bayesian hyperparameter search, not a manual grid.

**Final results, same test split:**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression (tuned) | 91.34% | 92.39% | 91.65% | 92.02% |
| DistilBERT (Optuna-tuned) | **94.01%** | 93.50% | **95.64%** | **94.56%** |

DistilBERT wins outright, and specifically on **recall** — the metric that matters most here, since a false negative (missing depressive language) is a more serious failure mode than a false positive in a mental-health screening context. The Logistic Regression baseline isn't obsolete despite losing on every metric, though: its coefficients are directly readable ("this word contributes +0.8 toward the depressive class"), while DistilBERT needs SHAP to get comparable — and more expensive to compute — interpretability.

---

## Ethical Considerations

- The dataset is publicly available and anonymized at the source.
- Predictions are probabilistic pattern-matches on text, not a clinical assessment — model outputs should never be treated as or substitute for a professional mental health evaluation.
- False negatives (missing depressive language) are the more serious failure mode in this context, which is why recall was weighted heavily in model comparison rather than optimizing for accuracy alone.