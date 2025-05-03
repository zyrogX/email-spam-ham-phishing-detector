#  Email Classification: Ham, Spam, and Phishing Detection

This project implements a machine learning pipeline to classify emails into three categories:
- **Ham** (legitimate emails),
- **Spam** (unsolicited marketing or junk), and
- **Phishing** (malicious attempts to steal information).

The pipeline uses:
- **TF-IDF** for feature extraction,
- **SMOTE** to address class imbalance, and
- **XGBoost** for multi-class classification.

Developed in a Jupyter Notebook with future plans for hyperparameter tuning and adversarial training.

---

##  Objectives

- Preprocess and combine Spam/Ham and Phishing datasets.
- Handle class imbalance using **SMOTE** and visualize distributions.
- Train and evaluate an **XGBoost** classifier for multi-class classification.
- Provide prediction functions with category and probability outputs.
- Save model and vectorizer for later use.
- Explore fine-tuning and adversarial robustness.

---

##  Dataset

This project uses two public datasets from **Kaggle**:

1. **Spam/Ham Dataset**: [Email Spam Detection Dataset](https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification)  
   - 5572 emails (Ham: `0`, Spam: `1`)

2. **Phishing Dataset**: [Phishing Email Data](https://www.kaggle.com/datasets/zyrogx/phishing)  
   - 189 emails (Phishing: `2`)

###  Class Distribution

| Class     | Initial | After SMOTE |
|-----------|---------|-------------|
| Ham       | 4825    | 4825        |
| Spam      | 747     | 4825        |
| Phishing  | 189     | 4825        |
| **Total** | 5761    | 14,475      |

---

##  ML Pipeline

Implemented in `spam-phishing-detection-zyrogx.ipynb`, the pipeline includes:

1. **Data Preprocessing**:
   - Combine datasets
   - Clean text and assign labels (Ham: 0, Spam: 1, Phishing: 2)

2. **Feature Extraction**:
   - Use `TfidfVectorizer` with 5000 features

3. **Balancing**:
   - Apply **SMOTE** to oversample Spam and Phishing classes

4. **Visualization**:
   - Plot class distribution before and after SMOTE

5. **Training**:
   - Train an **XGBoost** classifier using `multi:softprob` objective

6. **Evaluation**:
   - Measure **precision**, **recall**, **F1-score**

7. **Prediction**:
   - Predict category and probabilities for new emails

8. **Model Saving**:
   - Save model (`email_classifier_model.pkl`)
   - Save vectorizer (`tfidf_vectorizer.pkl`)

---

## 🗂️ Files

- `spam-phishing-detection-zyrogx.ipynb`: Main notebook
- `email_classifier_model.pkl`: Trained XGBoost model  
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer  

---

## ⚙️ Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
