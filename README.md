# Sentiment Analysis on IMDB Movie Reviews

This project implements a **sentiment analysis model** to classify IMDB movie reviews as positive or negative using Python and machine learning techniques. The dataset is sourced from **Hugging Face Datasets**.

---

## 📌 Project Steps

### 1. **Data Loading**
- The IMDB dataset is loaded using the `datasets` library from Hugging Face.

### 2. **Text Preprocessing**
- Special characters and punctuation are removed.
- Text is converted to lowercase.
- Tokenization is applied to split the text into individual words.
- Stopwords (common words like "the", "is", etc.) are removed.
- Lemmatization reduces words to their base form (e.g., "running" to "run").

### 3. **Feature Engineering**
- Text data is converted into numerical features using **TF-IDF Vectorization** with the top 5000 most important words.

### 4. **Model Training**
- A **Logistic Regression** model is trained on 80% of the dataset.

### 5. **Model Evaluation**
- The model is tested on the remaining 20% and evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

### 6. **Prediction Function**
- A helper function `predict_sentiment(text)` allows you to input custom reviews and get sentiment predictions.

---

## 🚀 How to Run the Script

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

### 2. Install Required Packages
Make sure you have Python 3.x installed.
```bash
pip install -r requirements.txt
```
Include the following in `requirements.txt`:
```txt
pandas
numpy
nltk
scikit-learn
datasets
```

### 3. Download NLTK Resources (if running for the first time)
These are automatically downloaded in the script, but you can do it manually:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Run the Script
```bash
python sentiment_analysis.py
```

---

## 🔍 Observations
- Logistic Regression performs reasonably well on the IMDB dataset.
- Preprocessing plays a critical role in improving model accuracy.
- TF-IDF is effective for text classification tasks with relatively short texts.
- Potential improvements: use word embeddings (e.g., Word2Vec or BERT) or try other classifiers like SVM or deep learning models.

---

## 📂 Files
- `sentiment_analysis.py` — Main Python script
- `README.md` — Project documentation
- `requirements.txt` — List of dependencies

---

## 📬 Contact
For questions or suggestions, feel free to open an issue or contact [urubaftb@gmail.com].

---

**Happy Learning! 🤖**

