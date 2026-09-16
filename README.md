# 🔍 Quora Duplicate Question Pairs

An end-to-end **Machine Learning + NLP project** that predicts whether two questions have the same meaning.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-2ea44f?style=for-the-badge)](https://quora-duplicate-questions-pairs-ayush45.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ayush4628/Quora-Duplicate-Question-Pairs)

---

## 📌 Overview

**Quora Duplicate Question Pairs** is a Natural Language Processing and Machine Learning project inspired by the Quora Question Pairs problem.

The application takes two questions as input and predicts whether they are:

- 🟢 **Duplicate Questions**
- 🔴 **Not Duplicate Questions**

The project covers the complete Machine Learning workflow:

**Data → Preprocessing → Feature Engineering → Text Representation → Model Training → Evaluation → Deployment**

---

## 🌐 Live Demo

🚀 **Live Application:**  
https://quora-duplicate-questions-pairs-ayush45.streamlit.app/

💻 **GitHub Repository:**  
https://github.com/ayush4628/Quora-Duplicate-Question-Pairs

The deployed application provides a simple interface where users can enter two questions and get the prediction along with the model confidence.

---

## 🔍 How It Works

```text
Question 1 + Question 2
          ↓
   Text Preprocessing
          ↓
   Feature Engineering
          ↓
   22 Handcrafted Features
          +
   Bag of Words Features
          ↓
     Truncated SVD
          ↓
    122 Final Features
          ↓
       XGBoost
          ↓
 Duplicate / Not Duplicate
          ↓
 Prediction + Confidence
```

---

# 📚 Dataset

The project uses a Quora-style question-pair dataset containing **404,351 question pairs**.

| Property | Details |
|---|---|
| Dataset | Quora Question Pairs |
| Total Rows | 404,351 |
| Target Column | `is_duplicate` |
| Classes | 2 |
| Problem Type | Binary Classification |

### Dataset Columns

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `qid1` | ID of Question 1 |
| `qid2` | ID of Question 2 |
| `question1` | First question |
| `question2` | Second question |
| `is_duplicate` | Target variable |

---

# 🧹 Text Preprocessing

Before creating features, both questions are cleaned and normalized.

### Main preprocessing steps

- Convert text to lowercase
- Remove leading and trailing spaces
- Normalize special symbols such as `%`, `$`, `₹`, `€`, and `@`
- Convert large number formats into compact representations
- Expand common contractions
- Remove HTML tags
- Remove punctuation
- Normalize whitespace

### Preprocessing Flow

```text
Raw Question
     ↓
Lowercase
     ↓
Special Character Handling
     ↓
Number Normalization
     ↓
Contraction Handling
     ↓
HTML Removal
     ↓
Punctuation Removal
     ↓
Clean Question
```

---

# ⚙️ Feature Engineering

The model uses **22 handcrafted features** to capture different types of similarity between two questions.

These features are divided into:

1. Basic Features
2. Token Features
3. Length Features
4. Fuzzy Similarity Features

---

## 1️⃣ Basic Features

| Feature | Description |
|---|---|
| `q1_len` | Character length of Question 1 |
| `q2_len` | Character length of Question 2 |
| `q1_num_words` | Number of words in Question 1 |
| `q2_num_words` | Number of words in Question 2 |
| `word_common` | Number of common unique words |
| `word_total` | Total number of unique words across both questions |
| `word_share` | Ratio of common words to total unique words |

---

## 2️⃣ Token Features

These features measure word, stop-word, and token overlap between the two questions.

| Feature | Description |
|---|---|
| `cwc_min` | Ratio of common words to the smaller question-word set |
| `cwc_max` | Ratio of common words to the larger question-word set |
| `csc_min` | Ratio of common stop words to the smaller stop-word count |
| `csc_max` | Ratio of common stop words to the larger stop-word count |
| `ctc_min` | Ratio of common tokens to the smaller token count |
| `ctc_max` | Ratio of common tokens to the larger token count |
| `last_word_eq` | `1` if the last word is the same, otherwise `0` |
| `first_word_eq` | `1` if the first word is the same, otherwise `0` |

---

## 3️⃣ Length Features

| Feature | Description |
|---|---|
| `abs_len_diff` | Absolute difference between the word counts of both questions |
| `mean_len` | Mean number of words in the two questions |
| `longest_substr_ratio` | Ratio of the longest common substring to the length of the smaller question |

---

## 4️⃣ Fuzzy Similarity Features

Fuzzy matching techniques are used to calculate textual similarity.

| Feature | Description |
|---|---|
| `fuzz_ratio` | Overall fuzzy similarity between the questions |
| `fuzz_partial_ratio` | Partial fuzzy similarity |
| `token_sort_ratio` | Similarity after sorting the tokens |
| `token_set_ratio` | Similarity based on unique token sets |

---

# 📝 Text Representation

## Bag of Words

`CountVectorizer` is used to convert the questions into numerical features.

The vectorizer uses:

```text
max_features = 3000
```

Each question is represented using up to 3,000 Bag of Words features.

```text
Question 1 → 3,000 features
Question 2 → 3,000 features

Combined → 6,000 features
```

---

## Truncated SVD

The 6,000 Bag of Words features are reduced using **Truncated SVD**.

```text
6,000 BoW Features
        ↓
Truncated SVD
        ↓
100 Features
```

This helps reduce the dimensionality of the text representation.

---

## Final Feature Vector

The final model input contains:

```text
22 Handcrafted Features
        +
100 SVD Features
        =
122 Final Features
```

So every question pair is finally represented using **122 features**.

---

# 🤖 Machine Learning Models

Four Machine Learning algorithms were trained and compared.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 81.18% | 74.23% | 75.10% | 74.66% |
| XGBoost | 80.22% | 73.30% | 73.06% | 73.18% |
| Linear SVM | 74.55% | 67.03% | 61.18% | 63.97% |
| Logistic Regression | 74.51% | 67.00% | 61.02% | 63.87% |

---

# 📊 Model Performance

## XGBoost

The deployed model is **XGBoost**.

| Metric | Score |
|---|---:|
| Accuracy | **80.22%** |
| Precision | **73.30%** |
| Recall | **73.06%** |
| F1 Score | **73.18%** |

### Why XGBoost was used for deployment

Random Forest produced slightly higher evaluation scores in the experiments, but the saved Random Forest model was much larger.

XGBoost provided competitive performance with a significantly smaller deployment artifact, making it more practical for the deployed application.

---

# 💾 Model Artifacts

The deployed application uses the following files:

```text
models/
├── duplicate_question_xgb.json
├── count_vectorizer.pkl
└── svd.pkl
```

### Purpose of each file

| File | Purpose |
|---|---|
| `duplicate_question_xgb.json` | Trained XGBoost classification model |
| `count_vectorizer.pkl` | Saved Bag of Words vectorizer |
| `svd.pkl` | Saved Truncated SVD transformer |

Keeping the same preprocessing and transformation pipeline during inference ensures that new questions are converted into the same feature representation used during training.

---

# 🛠️ Tech Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn
- XGBoost

### NLP

- NLTK
- FuzzyWuzzy

### Text Processing

- CountVectorizer
- Truncated SVD

### Deployment

- Streamlit
- Streamlit Community Cloud

### Development & Version Control

- Jupyter Notebook
- Git
- GitHub

---

# ✨ Application Features

- Enter Question 1 and Question 2
- Predict whether the questions are duplicates
- Display prediction confidence
- NLP-based text preprocessing
- 22 handcrafted similarity features
- Bag of Words text representation
- Truncated SVD dimensionality reduction
- XGBoost classification
- Real-time prediction through Streamlit
- Simple and clean user interface

---

# 📂 Project Structure

```text
Quora-Duplicate-Question-Pairs/
│
├── models/
│   ├── duplicate_question_xgb.json
│   ├── count_vectorizer.pkl
│   └── svd.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── ...
```

---

# 💻 Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/ayush4628/Quora-Duplicate-Question-Pairs.git
cd Quora-Duplicate-Question-Pairs
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🎯 Example Predictions

## 🟢 Duplicate Example

```text
Question 1:
How can I learn Python?

Question 2:
What is the best way to learn Python?
```

**Expected Result:** Duplicate Questions

---

## 🔴 Not Duplicate Example

```text
Question 1:
How can I learn Python?

Question 2:
How can I learn Java?
```

**Expected Result:** Not Duplicate Questions

---

## 🟢 Duplicate Example 2

```text
Question 1:
What is the capital of India?

Question 2:
Which city is the capital of India?
```

**Expected Result:** Duplicate Questions

---

## 🔴 Not Duplicate Example 2

```text
Question 1:
What is the capital of India?

Question 2:
What is the population of India?
```

**Expected Result:** Not Duplicate Questions

---

# 📈 Evaluation Metrics

The project evaluates the classification models using:

### Accuracy

Percentage of total predictions that are correct.

### Precision

Measures how many questions predicted as duplicate are actually duplicate.

### Recall

Measures how many actual duplicate question pairs are correctly identified.

### F1 Score

Harmonic mean of Precision and Recall.

### Confusion Matrix

Shows:

- True Positive
- True Negative
- False Positive
- False Negative

---

# 📚 Key Learning Outcomes

Through this project, I worked on the complete NLP and Machine Learning pipeline.

## NLP

- Text preprocessing
- Stopword handling
- Tokenization
- Token overlap
- Fuzzy similarity
- Bag of Words
- Dimensionality reduction
- Feature engineering

## Machine Learning

- Binary classification
- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Model serialization

## Deployment

- Saving trained ML artifacts
- Loading models during inference
- Building a Streamlit application
- Maintaining the same preprocessing pipeline during inference
- Deploying a Machine Learning application

---

# 🚀 Future Improvements

- [ ] Add TF-IDF features
- [ ] Add semantic embeddings
- [ ] Experiment with transformer-based models
- [ ] Improve semantic similarity detection
- [ ] Add model explainability
- [ ] Add prediction history
- [ ] Add probability visualizations
- [ ] Add automated testing
- [ ] Improve application performance

---

# 📊 Project Status

| Component | Status |
|---|---|
| Dataset | ✅ Completed |
| EDA | ✅ Completed |
| Text Preprocessing | ✅ Completed |
| Feature Engineering | ✅ Completed |
| Bag of Words | ✅ Completed |
| Truncated SVD | ✅ Completed |
| Model Training | ✅ Completed |
| Model Comparison | ✅ Completed |
| XGBoost Deployment Model | ✅ Completed |
| Streamlit Application | ✅ Completed |
| Cloud Deployment | ✅ Completed |
| GitHub Repository | ✅ Completed |

---

# 👨‍💻 Author

## Ayush Maurya

**Data Science / Machine Learning Enthusiast**

### Areas of Interest

- Python
- Data Science
- Machine Learning
- Deep Learning
- Natural Language Processing
- Generative AI
- Machine Learning Deployment

### 🔗 Connect With Me

- **GitHub:** https://github.com/ayush4628
- **LinkedIn:** https://www.linkedin.com/in/ayush4628/

---

# 📄 License

This project is licensed under the **MIT License**.

---

# ⭐ Support

If you found this project useful, consider giving the repository a ⭐ **Star** on GitHub.

---

<p align="center">

**Built with Python, NLP, Machine Learning & curiosity.**

**Made with ❤️ by Ayush**

</p>
