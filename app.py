import streamlit as st
import numpy as np
import pandas as pd
import re
import joblib
import nltk

from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
from scipy.sparse import hstack
from xgboost import XGBClassifier
from datetime import datetime


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)


# =========================================================
# Custom CSS
# =========================================================

st.html("""
<style>

    /* =====================================================
       Main Container
    ===================================================== */

    .block-container {
        max-width: 800px;
        padding-top: 1rem;
        padding-bottom: 0.3rem;
    }


    /* =====================================================
       Main Header
    ===================================================== */

    .custom-title {
        text-align: center;
        font-size: 2.15rem;
        font-weight: 750;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }


    .custom-subtitle {
        text-align: center;
        font-size: 0.82rem;
        opacity: 0.70;
        margin-bottom: 1rem;
    }


    /* =====================================================
       Information Card
    ===================================================== */

    .info-card {
        padding: 0.7rem 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.06);
        text-align: center;
        margin-bottom: 0.9rem;
        line-height: 1.4;
        font-size: 0.82rem;
    }


    .info-title {
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }


    /* =====================================================
       Question Labels
    ===================================================== */

    .question-label {
        font-size: 0.9rem;
        font-weight: 650;
        margin-top: 0.25rem;
        margin-bottom: 0.2rem;
    }


    /* =====================================================
       Text Areas
    ===================================================== */

    textarea {
        border-radius: 8px !important;
    }


    /* =====================================================
       Result Card
    ===================================================== */

    .result-card {
        padding: 0.9rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.05);
        text-align: center;
        margin-top: 0.5rem;
    }


    .result-title {
        font-size: 1.15rem;
        font-weight: 700;
    }


    .result-confidence {
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }


    /* =====================================================
       Footer
    ===================================================== */

    .custom-footer {
        margin-top: 1.2rem;
        padding-top: 0.4rem;
        padding-bottom: 0.2rem;
        text-align: center;
        opacity: 0.65;
        font-size: 0.75rem;
        line-height: 1.3;
    }


    .footer-line {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.35rem;
    }


    .footer-line::before,
    .footer-line::after {
        content: "";
        height: 1px;
        width: 100px;
        background: rgba(128, 128, 128, 0.5);
    }


    .footer-heart {
        margin: 0 10px;
        font-size: 0.8rem;
    }

</style>
""")


# =========================================================
# Load Required Files
# =========================================================

@st.cache_resource
def load_models():

    # -----------------------------------------------------
    # Load XGBoost Model
    # -----------------------------------------------------

    model = XGBClassifier()

    model.load_model(
        "models/duplicate_question_xgb.json"
    )


    # -----------------------------------------------------
    # Load CountVectorizer
    # -----------------------------------------------------

    cv = joblib.load(
        "models/count_vectorizer.pkl"
    )


    # -----------------------------------------------------
    # Load SVD
    # -----------------------------------------------------

    svd = joblib.load(
        "models/svd.pkl"
    )


    return model, cv, svd


model, cv, svd = load_models()


# =========================================================
# NLTK Stopwords
# =========================================================

nltk.download(
    "stopwords",
    quiet=True
)

from nltk.corpus import stopwords

STOP_WORDS = set(
    stopwords.words("english")
)


# =========================================================
# Contractions
# =========================================================

contractions = {

    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",

    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",

    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",

    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",

    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",

    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",

    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",

    "isn't": "is not",

    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",

    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",

    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",

    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",

    "needn't": "need not",
    "o'clock": "of the clock",

    "oughtn't": "ought not",
    "oughtn't've": "ought not have",

    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",

    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",

    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",

    "so've": "so have",
    "so's": "so as",

    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",

    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",

    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",

    "to've": "to have",

    "wasn't": "was not",

    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",

    "weren't": "were not",

    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",

    "when's": "when is",
    "when've": "when have",

    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",

    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",

    "why's": "why is",
    "why've": "why have",

    "will've": "will have",

    "won't": "will not",
    "won't've": "will not have",

    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",

    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",

    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}


# =========================================================
# Preprocessing
# =========================================================

def preprocess(q):

    q = str(q).lower().strip()


    # -----------------------------------------------------
    # Special Characters
    # -----------------------------------------------------

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')


    # -----------------------------------------------------
    # Math Tag
    # -----------------------------------------------------

    q = q.replace('[math]', '')


    # -----------------------------------------------------
    # Number Replacements
    # -----------------------------------------------------

    q = q.replace(
        ',000,000,000 ',
        'b '
    )

    q = q.replace(
        ',000,000 ',
        'm '
    )

    q = q.replace(
        ',000 ',
        'k '
    )


    q = re.sub(
        r'([0-9]+)000000000',
        r'\1b',
        q
    )

    q = re.sub(
        r'([0-9]+)000000',
        r'\1m',
        q
    )

    q = re.sub(
        r'([0-9]+)000',
        r'\1k',
        q
    )


    # -----------------------------------------------------
    # Decontract
    # -----------------------------------------------------

    q_decontracted = []

    for word in q.split():

        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)


    q = q.replace(
        "'ve",
        " have"
    )

    q = q.replace(
        "n't",
        " not"
    )

    q = q.replace(
        "'re",
        " are"
    )

    q = q.replace(
        "'ll",
        " will"
    )


    # -----------------------------------------------------
    # Remove HTML
    # -----------------------------------------------------

    q = BeautifulSoup(
        q,
        "html.parser"
    ).get_text()


    # -----------------------------------------------------
    # Remove Punctuation
    # -----------------------------------------------------

    pattern = re.compile(
        r'\W'
    )

    q = re.sub(
        pattern,
        ' ',
        q
    ).strip()


    return q


# =========================================================
# Token Features
# =========================================================

def fetch_token_features(q1, q2):

    SAFE_DIV = 0.0001

    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()


    if (
        len(q1_tokens) == 0
        or
        len(q2_tokens) == 0
    ):

        return token_features


    # -----------------------------------------------------
    # Non-stopwords
    # -----------------------------------------------------

    q1_words = set(
        word
        for word in q1_tokens
        if word not in STOP_WORDS
    )

    q2_words = set(
        word
        for word in q2_tokens
        if word not in STOP_WORDS
    )


    # -----------------------------------------------------
    # Stopwords
    # -----------------------------------------------------

    q1_stops = set(
        word
        for word in q1_tokens
        if word in STOP_WORDS
    )

    q2_stops = set(
        word
        for word in q2_tokens
        if word in STOP_WORDS
    )


    # -----------------------------------------------------
    # Common Words
    # -----------------------------------------------------

    common_word_count = len(
        q1_words.intersection(
            q2_words
        )
    )


    # -----------------------------------------------------
    # Common Stopwords
    # -----------------------------------------------------

    common_stop_count = len(
        q1_stops.intersection(
            q2_stops
        )
    )


    # -----------------------------------------------------
    # Common Tokens
    # -----------------------------------------------------

    common_token_count = len(
        set(q1_tokens).intersection(
            set(q2_tokens)
        )
    )


    # -----------------------------------------------------
    # Common Non-stopwords
    # -----------------------------------------------------

    token_features[0] = (
        common_word_count /
        (
            min(
                len(q1_words),
                len(q2_words)
            )
            +
            SAFE_DIV
        )
    )


    token_features[1] = (
        common_word_count /
        (
            max(
                len(q1_words),
                len(q2_words)
            )
            +
            SAFE_DIV
        )
    )


    # -----------------------------------------------------
    # Common Stopwords
    # -----------------------------------------------------

    token_features[2] = (
        common_stop_count /
        (
            min(
                len(q1_stops),
                len(q2_stops)
            )
            +
            SAFE_DIV
        )
    )


    token_features[3] = (
        common_stop_count /
        (
            max(
                len(q1_stops),
                len(q2_stops)
            )
            +
            SAFE_DIV
        )
    )


    # -----------------------------------------------------
    # Common Tokens
    # -----------------------------------------------------

    token_features[4] = (
        common_token_count /
        (
            min(
                len(q1_tokens),
                len(q2_tokens)
            )
            +
            SAFE_DIV
        )
    )


    token_features[5] = (
        common_token_count /
        (
            max(
                len(q1_tokens),
                len(q2_tokens)
            )
            +
            SAFE_DIV
        )
    )


    # -----------------------------------------------------
    # Last Word
    # -----------------------------------------------------

    token_features[6] = int(
        q1_tokens[-1] ==
        q2_tokens[-1]
    )


    # -----------------------------------------------------
    # First Word
    # -----------------------------------------------------

    token_features[7] = int(
        q1_tokens[0] ==
        q2_tokens[0]
    )


    return token_features


# =========================================================
# Length Features
# =========================================================

def fetch_length_features(q1, q2):

    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()


    if (
        len(q1_tokens) == 0
        or
        len(q2_tokens) == 0
    ):

        return length_features


    # -----------------------------------------------------
    # Absolute Length Difference
    # -----------------------------------------------------

    length_features[0] = abs(
        len(q1_tokens) -
        len(q2_tokens)
    )


    # -----------------------------------------------------
    # Mean Length
    # -----------------------------------------------------

    length_features[1] = (
        len(q1_tokens) +
        len(q2_tokens)
    ) / 2


    # -----------------------------------------------------
    # Longest Common Substring
    # -----------------------------------------------------

    strs = list(
        distance.lcsubstrings(
            q1,
            q2
        )
    )


    if len(strs) > 0:

        length_features[2] = (
            len(strs[0]) /
            (
                min(
                    len(q1),
                    len(q2)
                )
                +
                1
            )
        )

    else:

        length_features[2] = 0.0


    return length_features


# =========================================================
# Fuzzy Features
# =========================================================

def fetch_fuzzy_features(q1, q2):

    fuzzy_features = [0.0] * 4


    fuzzy_features[0] = fuzz.QRatio(
        q1,
        q2
    )


    fuzzy_features[1] = fuzz.partial_ratio(
        q1,
        q2
    )


    fuzzy_features[2] = fuzz.token_sort_ratio(
        q1,
        q2
    )


    fuzzy_features[3] = fuzz.token_set_ratio(
        q1,
        q2
    )


    return fuzzy_features


# =========================================================
# Create 22 Handcrafted Features
# =========================================================

def create_handcrafted_features(q1, q2):

    # -----------------------------------------------------
    # Basic Features
    # -----------------------------------------------------

    q1_len = len(q1)
    q2_len = len(q2)


    q1_num_words = len(
        q1.split(" ")
    )

    q2_num_words = len(
        q2.split(" ")
    )


    # -----------------------------------------------------
    # Common Words
    # -----------------------------------------------------

    w1 = set(
        map(
            lambda word:
            word.lower().strip(),
            q1.split(" ")
        )
    )

    w2 = set(
        map(
            lambda word:
            word.lower().strip(),
            q2.split(" ")
        )
    )


    word_common = len(
        w1 & w2
    )


    word_total = (
        len(w1) +
        len(w2)
    )


    word_share = (
        round(
            word_common /
            word_total,
            2
        )
        if word_total != 0
        else 0
    )


    # -----------------------------------------------------
    # Token Features
    # -----------------------------------------------------

    token_features = fetch_token_features(
        q1,
        q2
    )


    # -----------------------------------------------------
    # Length Features
    # -----------------------------------------------------

    length_features = fetch_length_features(
        q1,
        q2
    )


    # -----------------------------------------------------
    # Fuzzy Features
    # -----------------------------------------------------

    fuzzy_features = fetch_fuzzy_features(
        q1,
        q2
    )


    # -----------------------------------------------------
    # EXACT SAME COLUMN ORDER AS TRAINING
    # -----------------------------------------------------

    features = [

        # Basic features

        q1_len,
        q2_len,

        q1_num_words,
        q2_num_words,

        word_common,
        word_total,
        word_share,


        # Token features

        token_features[0],
        token_features[1],
        token_features[2],
        token_features[3],
        token_features[4],
        token_features[5],
        token_features[6],
        token_features[7],


        # Length features

        length_features[0],
        length_features[1],
        length_features[2],


        # Fuzzy features

        fuzzy_features[0],
        fuzzy_features[1],
        fuzzy_features[2],
        fuzzy_features[3]

    ]


    return np.array(
        features,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


# =========================================================
# Generate Final 122 Features
# =========================================================

def create_final_features(
    question1,
    question2
):

    # -----------------------------------------------------
    # Preprocess Questions
    # -----------------------------------------------------

    q1 = preprocess(
        question1
    )

    q2 = preprocess(
        question2
    )


    # -----------------------------------------------------
    # 22 Handcrafted Features
    # -----------------------------------------------------

    handcrafted = create_handcrafted_features(
        q1,
        q2
    )


    # -----------------------------------------------------
    # Bag of Words
    # -----------------------------------------------------

    questions = [
        q1,
        q2
    ]


    question_matrix = cv.transform(
        questions
    )


    # -----------------------------------------------------
    # Q1 and Q2 Separately
    # -----------------------------------------------------

    q1_arr = question_matrix[0]

    q2_arr = question_matrix[1]


    # -----------------------------------------------------
    # Combine Q1 + Q2
    # -----------------------------------------------------

    X_bow = hstack(
        [
            q1_arr,
            q2_arr
        ],
        format="csr"
    )


    # -----------------------------------------------------
    # SVD → 100 Features
    # -----------------------------------------------------

    X_bow_reduced = svd.transform(
        X_bow
    )


    # -----------------------------------------------------
    # 22 + 100 = 122 Features
    # -----------------------------------------------------

    final_features = np.hstack(
        [
            handcrafted,
            X_bow_reduced
        ]
    )


    return final_features.astype(
        np.float32
    )


# =========================================================
# STREAMLIT UI
# =========================================================


# =========================================================
# Main Header
# =========================================================

st.html("""
<div class="custom-title">
    🔍 Duplicate Question Detector
</div>

<div class="custom-subtitle">
    AI-powered NLP system to determine whether two questions
    have the same meaning.
</div>
""")


# =========================================================
# Information Card
# =========================================================

st.html("""
<div class="info-card">

    <div class="info-title">
        How it works?
    </div>

    Enter two questions below and let the trained
    Machine Learning model determine whether they
    are duplicate questions.

</div>
""")


# =========================================================
# Question 1
# =========================================================

st.html("""
<div class="question-label">
    Question 1
</div>
""")


question1 = st.text_area(
    "Question 1",
    placeholder="e.g. What is the best way to learn Python?",
    height=85,
    label_visibility="collapsed"
)


# =========================================================
# Question 2
# =========================================================

st.html("""
<div class="question-label">
    Question 2
</div>
""")


question2 = st.text_area(
    "Question 2",
    placeholder="e.g. How can I learn Python effectively?",
    height=85,
    label_visibility="collapsed"
)


# =========================================================
# Prediction Button
# =========================================================

st.write("")


if st.button(
    "🔎  Check for Duplicate",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # Validate Input
    # -----------------------------------------------------

    if (
        not question1.strip()
        or
        not question2.strip()
    ):

        st.warning(
            "⚠️ Please enter both questions."
        )


    else:

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        with st.spinner(
            "🤖 Analyzing questions..."
        ):

            # Create 122 features

            X_input = create_final_features(
                question1,
                question2
            )


            # Model Prediction

            prediction = model.predict(
                X_input
            )[0]


            # Prediction Probability

            probability = model.predict_proba(
                X_input
            )[0]


            duplicate_probability = probability[1]


        # =================================================
        # Result
        # =================================================

        st.divider()


        # -------------------------------------------------
        # Duplicate
        # -------------------------------------------------

        if prediction == 1:

            confidence = (
                duplicate_probability *
                100
            )


            st.html(f"""
            <div class="result-card">

                <div class="result-title">
                    🟢 Duplicate Questions
                </div>

                <div class="result-confidence">
                    Model Confidence:
                    <strong>{confidence:.2f}%</strong>
                </div>

            </div>
            """)


        # -------------------------------------------------
        # Not Duplicate
        # -------------------------------------------------

        else:

            confidence = (
                (1 - duplicate_probability) *
                100
            )


            st.html(f"""
            <div class="result-card">

                <div class="result-title">
                    🔴 Not Duplicate Questions
                </div>

                <div class="result-confidence">
                    Model Confidence:
                    <strong>{confidence:.2f}%</strong>
                </div>

            </div>
            """)


# =========================================================
# Footer
# =========================================================

current_year = datetime.now().year


st.html(f"""
<div class="custom-footer">

    <div class="footer-line">
        <span class="footer-heart">❤️</span>
    </div>

    Built with ⚡ <strong>Streamlit</strong>
    &nbsp; • &nbsp;
    Made with ❤️ by <strong>Ayush</strong>
    &nbsp; • &nbsp;
    © {current_year}

</div>
""")