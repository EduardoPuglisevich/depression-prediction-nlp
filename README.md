# Depression Prediction using NLP

## Overview
This project focuses on the automatic detection of language associated with depression in Reddit posts using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to evaluate how effectively textual patterns can be used to classify posts as depressive or non-depressive, comparing traditional models with transformer-based architectures.

The project combines interpretability-focused classical models with state-of-the-art contextual language models, aiming to balance performance, explainability, and practical applicability.

---

## Problem Statement
Depression is one of the most prevalent mental health disorders worldwide, yet access to timely diagnosis and treatment remains limited. Social media platforms such as Reddit contain large volumes of spontaneous textual expressions that may reveal early indicators of emotional distress.

This project addresses the following research questions:
- Can machine learning models accurately classify Reddit posts as depressive or non-depressive?
- Which textual features contribute most to depressive language detection?
- How do traditional NLP models compare against transformer-based approaches?

---

## Dataset
The project uses the **Reddit Depression Dataset** (Kaggle), consisting of user-generated posts and comments related to mental health topics.

- Original size: ~2.47M records
- Final working dataset: ~78K balanced samples
- Labels:  
  - `1`: Depressive language  
  - `0`: Non-depressive language  

Two text representations were used:
- `body`: normalized text for traditional ML models
- `body_raw`: original text preserved for transformer-based models

---

## Methodology
1. Data cleaning and balancing
2. Text preprocessing (tokenization, stopword removal, stemming)
3. Feature engineering (word counts, lexical diversity, frequency metrics)
4. Model training and evaluation using a stratified 80/20 split
5. Hyperparameter optimization
6. Model interpretability analysis

---

## Models
Two main approaches were implemented:

### Logistic Regression + TF-IDF
- N-grams: unigrams and bigrams
- Vocabulary size: 20,000
- Focus on interpretability via feature coefficients

### DistilBERT (Transformer-based)
- Fine-tuned on raw Reddit text
- Optimized using Optuna
- Interpretability supported via SHAP analysis

---

## Results
| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| Logistic Regression (Tuned) | 91.34% | 92.39% | 91.65% | 92.02% |
| DistilBERT (Optuna Tuned) | **94.01%** | 93.50% | **95.64%** | **94.56%** |

The transformer-based model achieved superior performance, particularly in Recall, which is critical for minimizing false negatives in mental health-related detection tasks.

---

## Deployment
A lightweight web interface was developed to allow real-time text classification using the trained models. Users can input text and receive:
- Predicted class (Depressive / Non-depressive)
- Probability score

---

## Ethical Considerations
- The dataset contains publicly available, anonymized data.
- Predictions are probabilistic and not intended for clinical diagnosis.
- Model outputs should not replace professional mental health evaluation.

---

## Tech Stack
- Python
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Optuna
- SHAP
- Pandas, NumPy, Matplotlib

---

## Contributors
- **Eduardo Elías Puglisevich Vergara**  
- Samuel Esteban Cano Chocce  
- Nicolás Miguel Guerrero Icochea  

