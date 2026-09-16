```{=html}
<p align="center">
```
# 🔍 Quora Duplicate Question Pairs

An end-to-end **Machine Learning and NLP application** that predicts
whether two questions have the same meaning.

```{=html}
</p>
```
```{=html}
<p align="center">
```
`<a href="https://quora-duplicate-questions-pairs-ayush45.streamlit.app/">`{=html}
`<img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-2ea44f?style=for-the-badge" alt="Live Demo">`{=html}
`</a>`{=html}

`<a href="https://github.com/ayush4628/Quora-Duplicate-Question-Pairs">`{=html}
`<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">`{=html}
`</a>`{=html}

```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Overview

**Quora Duplicate Question Pairs** is an end-to-end **Natural Language
Processing and Machine Learning** project that predicts whether two
questions are semantically duplicate.

The application takes two questions as input and classifies them into:

-   🟢 **Duplicate Questions**
-   🔴 **Not Duplicate Questions**

The project covers the complete workflow from **data preprocessing and
exploratory data analysis to feature engineering, model training,
evaluation, model serialization, and Streamlit deployment**.

------------------------------------------------------------------------

## 🌐 Live Demo

**Live Project:**\
https://quora-duplicate-questions-pairs-ayush45.streamlit.app/

**GitHub Repository:**\
https://github.com/ayush4628/Quora-Duplicate-Question-Pairs

Enter two questions and click **Check for Duplicate** to see the
prediction and confidence score.

### Example

``` text
Question 1:
How can I learn Python?

Question 2:
What is the best way to learn Python?

Prediction:
Duplicate Questions
```

------------------------------------------------------------------------

## 🔍 How It Works

``` text
Two Questions
      ↓
Text Preprocessing
      ↓
Feature Engineering
      ↓
22 Handcrafted Features
      +
Bag of Words
      ↓
Truncated SVD
      ↓
122 Final Features
      ↓
XGBoost Model
      ↓
Prediction Probability
      ↓
Duplicate / Not Duplicate
```

------------------------------------------------------------------------

## 📚 Dataset

The project uses a Quora-style question-pair dataset.

  Property     Details
  ------------ -----------------------
  Dataset      Quora Question Pairs
  Total Rows   404,351
  Target       `is_duplicate`
  Classes      2
  Task         Binary Classification

### Main Columns

  Column           Description
  ---------------- ---------------------------
  `id`             Unique row identifier
  `qid1`           ID of the first question
  `qid2`           ID of the second question
  `question1`      First question
  `question2`      Second question
  `is_duplicate`   Target variable

------------------------------------------------------------------------

## 🧹 Text Preprocessing

The questions go through several preprocessing steps before feature
extraction.

### Main Steps

-   Convert text to lowercase
-   Remove leading and trailing spaces
-   Normalize `%`, `$`, `₹`, `€`, and `@`
-   Handle large number formats
-   Expand common contractions
-   Remove HTML
-   Remove punctuation
-   Normalize the final text

``` text
Raw Question
      ↓
Lowercase
      ↓
Special Character Handling
      ↓
Contraction Handling
      ↓
HTML Removal
      ↓
Punctuation Removal
      ↓
Clean Question
```

------------------------------------------------------------------------

# ⚙️ Feature Engineering

The model uses **22 handcrafted features** to capture lexical overlap,
token similarity, question length, and fuzzy similarity.

------------------------------------------------------------------------

## 1. Basic Features

  Feature          Description
  ---------------- ----------------------------------------------------
  `q1_len`         Character length of Question 1
  `q2_len`         Character length of Question 2
  `q1_num_words`   Number of words in Question 1
  `q2_num_words`   Number of words in Question 2
  `word_common`    Number of common unique words
  `word_total`     Total number of unique words across both questions
  `word_share`     Ratio of common words to total unique words

------------------------------------------------------------------------

## 2. Token Features

These features measure the overlap between words and tokens in both
questions.

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `cwc_min`                           Ratio of common words to the length
                                      of the smaller question-word set

  `cwc_max`                           Ratio of common words to the length
                                      of the larger question-word set

  `csc_min`                           Ratio of common stop words to the
                                      smaller stop-word count

  `csc_max`                           Ratio of common stop words to the
                                      larger stop-word count

  `ctc_min`                           Ratio of common tokens to the
                                      smaller token count

  `ctc_max`                           Ratio of common tokens to the
                                      larger token count

  `last_word_eq`                      `1` if the last word is the same,
                                      otherwise `0`

  `first_word_eq`                     `1` if the first word is the same,
                                      otherwise `0`
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 3. Length-Based Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `abs_len_diff`                      Absolute difference between the
                                      number of words in the two
                                      questions

  `mean_len`                          Mean number of words in the two
                                      questions

  `longest_substr_ratio`              Ratio of the longest common
                                      substring to the length of the
                                      smaller question
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Fuzzy Features

These features measure textual similarity using **FuzzyWuzzy**.

  Feature                Description
  ---------------------- ---------------------------------------
  `fuzz_ratio`           Overall fuzzy similarity score
  `fuzz_partial_ratio`   Partial fuzzy similarity score
  `token_sort_ratio`     Similarity after sorting tokens
  `token_set_ratio`      Similarity based on unique token sets

------------------------------------------------------------------------

## 📝 Text Representation

### Bag of Words

`CountVectorizer` is used with a maximum of **3,000 features** for each
question.

``` text
Question 1 → 3,000 features
Question 2 → 3,000 features
```

These are combined:

``` text
3,000 + 3,000 = 6,000 BoW features
```

### Truncated SVD

To reduce dimensionality:

``` text
6,000 BoW features
        ↓
Truncated SVD
        ↓
100 features
```

### Final Feature Vector

``` text
22 Handcrafted Features
        +
100 SVD Features
        =
122 Final Features
```

------------------------------------------------------------------------

## 🤖 Machine Learning Models

Multiple Machine Learning algorithms were experimented with and
compared.

  Model                   Accuracy   Precision   Recall   F1 Score
  --------------------- ---------- ----------- -------- ----------
  Random Forest             81.18%      74.23%   75.10%     74.66%
  XGBoost                   80.22%      73.30%   73.06%     73.18%
  Linear SVM                74.55%      67.03%   61.18%     63.97%
  Logistic Regression       74.51%      67.00%   61.02%     63.87%

The final deployed application uses **XGBoost** because it provided
competitive performance with a much smaller deployment artifact than the
Random Forest model.

------------------------------------------------------------------------

## 📊 Final Model Performance

### XGBoost

  Metric             Score
  ----------- ------------
  Accuracy      **80.22%**
  Precision     **73.30%**
  Recall        **73.06%**
  F1 Score      **73.18%**

------------------------------------------------------------------------

## 🧠 Why XGBoost for Deployment?

Random Forest achieved slightly higher test accuracy during
experimentation, but its saved model was comparatively large.

XGBoost was selected for deployment because it provided competitive
performance while keeping the deployment model significantly smaller.

### Deployment Artifacts

``` text
duplicate_question_xgb.json
count_vectorizer.pkl
svd.pkl
```

The XGBoost model is stored using its native model format for
deployment.

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Category                   Technology
  -------------------------- -------------------------------
  Programming                Python
  Machine Learning           Scikit-learn
  Gradient Boosting          XGBoost
  NLP                        NLTK
  Text Vectorization         CountVectorizer
  Dimensionality Reduction   Truncated SVD
  Text Similarity            FuzzyWuzzy
  Data Processing            NumPy / Pandas
  Sparse Matrices            SciPy
  Web Application            Streamlit
  Model Serialization        Joblib / XGBoost native model
  Development                Jupyter Notebook
  Version Control            Git / GitHub
  Deployment                 Streamlit Community Cloud

------------------------------------------------------------------------

## ✨ Application Features

-   Enter two questions through a simple web interface
-   Predict duplicate or non-duplicate questions
-   Display model confidence
-   NLP-based preprocessing
-   22 handcrafted similarity features
-   Bag of Words representation
-   SVD dimensionality reduction
-   XGBoost classification
-   Real-time prediction
-   Streamlit web interface

------------------------------------------------------------------------

## 📂 Project Structure

``` text
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

------------------------------------------------------------------------

## 💻 Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/ayush4628/Quora-Duplicate-Question-Pairs.git
cd Quora-Duplicate-Question-Pairs
```

### 2. Create a virtual environment

``` bash
python -m venv venv
```

### 3. Activate the environment

#### Windows

``` bash
venv\Scripts\activate
```

#### macOS / Linux

``` bash
source venv/bin/activate
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🎯 Example Predictions

### 🟢 Duplicate

``` text
Q1: How can I learn Python?

Q2: What is the best way to learn Python?
```

**Expected:** Duplicate Questions

### 🔴 Not Duplicate

``` text
Q1: How can I learn Python?

Q2: How can I learn Java?
```

**Expected:** Not Duplicate Questions

### 🟢 Another Duplicate Example

``` text
Q1: What is the capital of India?

Q2: Which city is the capital of India?
```

**Expected:** Duplicate Questions

### 🔴 Another Non-Duplicate Example

``` text
Q1: What is the capital of India?

Q2: What is the population of India?
```

**Expected:** Not Duplicate Questions

------------------------------------------------------------------------

## 📈 Evaluation Metrics

The project uses:

-   **Accuracy** --- Percentage of correctly classified question pairs.
-   **Precision** --- How many predicted duplicate pairs were actually
    duplicate.
-   **Recall** --- How many actual duplicate pairs were correctly
    identified.
-   **F1 Score** --- Harmonic mean of Precision and Recall.
-   **Confusion Matrix** --- Analysis of True Positives, True Negatives,
    False Positives and False Negatives.

------------------------------------------------------------------------

## 📚 Key Learning Outcomes

### Machine Learning

-   Classification algorithms
-   Model comparison
-   Hyperparameter tuning
-   Cross-validation
-   Model evaluation
-   Model serialization

### NLP

-   Text preprocessing
-   Stopword handling
-   Token-based similarity
-   Fuzzy text similarity
-   Bag of Words
-   Feature engineering
-   Dimensionality reduction

### Deployment

-   Saving trained ML artifacts
-   Loading models during inference
-   Building a Streamlit application
-   Keeping preprocessing consistent between training and inference
-   Deploying a Machine Learning application to the cloud

------------------------------------------------------------------------

## 🚀 Future Improvements

-   [ ] Add semantic embeddings
-   [ ] Experiment with TF-IDF features
-   [ ] Compare additional ML algorithms
-   [ ] Improve handling of semantically similar but differently worded
    questions
-   [ ] Add probability visualisation
-   [ ] Add prediction history
-   [ ] Add model explanation
-   [ ] Add automated testing
-   [ ] Improve deployment performance

------------------------------------------------------------------------

## 📊 Project Status

  Component                  Status
  -------------------------- --------------
  Dataset                    ✅ Completed
  EDA                        ✅ Completed
  Text Preprocessing         ✅ Completed
  Feature Engineering        ✅ Completed
  BoW + SVD                  ✅ Completed
  Model Training             ✅ Completed
  Model Comparison           ✅ Completed
  XGBoost Deployment Model   ✅ Completed
  Streamlit Application      ✅ Completed
  Cloud Deployment           ✅ Completed
  GitHub Repository          ✅ Completed

------------------------------------------------------------------------

## 👨‍💻 Author

### Ayush Maurya

**Data Science / Machine Learning Enthusiast**

Interested in:

-   Python
-   Data Science
-   Machine Learning
-   Deep Learning
-   Natural Language Processing
-   Generative AI
-   Machine Learning Deployment

### 🔗 Connect With Me

**GitHub:**\
https://github.com/ayush4628

**LinkedIn:**\
https://www.linkedin.com/in/ayush4628/

------------------------------------------------------------------------

## 📄 License

This project is licensed under the **MIT License**.

------------------------------------------------------------------------

## ⭐ Support

If you found this project useful or interesting, consider giving the
repository a **⭐ Star** on GitHub.

```{=html}
<p align="center">
```
**🔍 + 🧠 + 📝 = 🚀**

`<br>`{=html}

`<sub>`{=html}Built with Python, NLP, Machine Learning &
curiosity.`</sub>`{=html}

```{=html}
</p>
```
```{=html}
<p align="center">
```
Made with ❤️ for learning, experimentation and real-world Machine
Learning.

```{=html}
</p>
```
