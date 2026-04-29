# Sentiment Analysis for Indonesia Digital Identity App (DUKCAPIL)

A production-style NLP project that analyzes public sentiment toward Indonesia's **Identitas Kependudukan Digital** mobile app based on Google Play Store reviews.

This project combines **Indonesian text preprocessing**, **lexicon-based weak labeling**, and **supervised machine learning benchmarking** to identify the best-performing sentiment classifier for real-world monitoring.

## Executive Summary
- Built a full sentiment analysis workflow on **23,313** user reviews.
- Applied Indonesian-specific NLP pipeline (normalization, stopword removal, stemming with Sastrawi).
- Generated sentiment labels (`positive`, `negative`, `neutral`) using lexicon scoring.
- Benchmarked **13 ML model variants** with TF-IDF features.
- Final model: **Logistic Regression + Optuna tuning**, with **94.12% test accuracy**.

## Business Value
This project helps public-service product teams and policymakers:
- Track user pain points and satisfaction trends at scale.
- Prioritize product fixes with evidence from citizen feedback.
- Detect sentiment shifts after app updates or policy changes.
- Build data-driven service quality improvements for digital government.

## Dataset
Source data comes from Google Play Store reviews of the DUKCAPIL app.

Files:
- `dataset/dataset-dukcapilApp.csv` (processed review dataset)
- `dataset/datasetRawVer-dukcapilApp.csv` (raw dataset version)
- `dataset/slangwords.txt` (slang normalization dictionary)

### Data Size and Label Distribution
- Total reviews: **23,313**
- Sentiment labels created in notebook using Indonesian sentiment lexicons:
  - `negative`: **12,280** (52.67%)
  - `positive`: **7,058** (30.27%)
  - `neutral`: **3,975** (17.05%)

## Methodology
### 1. Text Preprocessing (Indonesian NLP)
- Cleaning mentions/URLs/numbers/punctuation
- Emoji removal
- Repeated-character normalization
- Case folding
- Tokenization (NLTK)
- Slang normalization (`slangwords.txt`)
- Stopword removal (Indonesian + custom additions)
- Stemming with **Sastrawi**

### 2. Weak Labeling via Lexicon Scoring
- Positive and negative lexicons loaded from external Indonesian lexicon resources.
- Polarity score computed per review.
- Labels assigned as:
  - `positive` if score > 0
  - `negative` if score < 0
  - `neutral` if score = 0

### 3. Feature Engineering
- TF-IDF vectorization (`max_features=200`, `min_df=17`, `max_df=0.8`)
- Train/test split: **80/20** (`random_state=42`)

### 4. Model Benchmarking
Compared multiple algorithms and tuning strategies:
- SVM (RBF)
- Random Forest
- Random Forest + Optuna
- XGBoost + Optuna
- CatBoost
- LightGBM
- Voting Classifier
- Ensemble Model
- Logistic Regression
- Logistic Regression + Optuna
- Gradient Boosting variants
- Decision Tree

## Model Performance (Test Accuracy)
From notebook outputs:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression + Optuna | **0.9412** |
| Logistic Regression | 0.8911 |
| Ensemble Model | 0.8889 |
| XGBoost + Optuna | 0.8883 |
| CatBoost | 0.8853 |
| SVM (RBF) | 0.8851 |
| Voting Classifier | 0.8840 |
| Gradient Boosting (GridSearchCV tuned) | 0.8803 |
| LightGBM | 0.8799 |
| Random Forest | 0.8769 |
| Random Forest + Optuna | 0.8600 |
| Gradient Boosting baseline | 0.8574 |
| Decision Tree | 0.8525 |

### Selected Final Model
**Logistic Regression with Optuna hyperparameter tuning** was selected as the main model due to the highest held-out test accuracy.

## Inference Demo
The notebook includes a prediction example for new user reviews using the selected model pipeline. This demonstrates how the model can be integrated into a monitoring workflow for live sentiment classification.

## Repository Structure
- `notebook.ipynb` - Main end-to-end analysis and modeling notebook.
- `scraping.ipynb` - Scraping workflow notebook.
- `scrapingRaw.ipynb` - Raw scraping workflow notebook.
- `dataset/` - Dataset files and slang dictionary.
- `requirements.txt` - Python dependencies.

## How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch notebook
```bash
jupyter notebook
```
Open `notebook.ipynb` and run cells sequentially.

## Tech Stack
- Python
- Pandas, NumPy
- NLTK, Sastrawi, Swifter
- Scikit-learn
- Optuna
- XGBoost, LightGBM, CatBoost
- Matplotlib, Seaborn, WordCloud

## Author
**Sulhan Fuadi**  
GitHub: https://github.com/sulhanfuadi
