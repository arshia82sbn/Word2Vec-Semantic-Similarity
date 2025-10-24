# Word2Vec Semantic Analysis and Text Preprocessing Project

## 📘 Introduction
This project focuses on **semantic word analysis** using **Word2Vec**, a powerful neural network-based model for representing words as continuous vectors in a high-dimensional space. By analyzing word similarities and relationships, we can extract meaningful insights from texts and visualize connections between concepts.

Our goal is to **build a pipeline** that performs text preprocessing, tokenization, lemmatization, and training a Word2Vec model on the processed data. We also compute similarity scores for selected words (like *love*, *friendship*, and *courage*) and visualize their relationships using charts.

---

## 🧠 Project Objectives
- Develop a **complete NLP pipeline** from raw text to processed tokens.
- Train a **Word2Vec model** for understanding semantic similarity.
- Explore how words like *love*, *friendship*, and *courage* relate in context.
- Visualize semantic similarities through **histograms and plots**.

---

## ⚙️ Code Breakdown and Explanation
Below is a detailed explanation of each part of the code, step by step.

---

### **Step 1: Import Required Libraries**
```python
import nltk
import re
import string
import pandas as pd
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
```
**Explanation:**
- `nltk` – used for natural language preprocessing (tokenization, stopwords, lemmatization).
- `re` – regular expressions for cleaning text.
- `string` – handles punctuation removal.
- `pandas` – data manipulation and dataframe creation.
- `plotly.express` – visualization library for interactive plots.
- `gensim.models.Word2Vec` – used to train our word embedding model.

---

### **Step 2: Download NLTK Resources**
```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
```
**Explanation:**
These commands download essential NLTK datasets for tokenization, lemmatization, and stopword filtering.

---

### **Step 3: Define Preprocessing Function**
```python
def preprocessing_text(text: str) -> list:
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = [re.sub(r'[^a-z\s]', '', word) for word in tokens]
    filtered_tokens = [word for word in clean_tokens if word not in stopwords and word != '']
    lemmatizer = WordNetLemmatizer()
    lemme_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemme_tokens
```
**Explanation:**
- Converts all text to lowercase.
- Tokenizes the sentence into words.
- Removes all non-alphabetic characters.
- Filters out stopwords (like *and*, *the*, *is*).
- Lemmatizes tokens to reduce them to their root form (e.g., *running → run*).

This function ensures clean and normalized text input for training the model.

---

### **Step 4: Apply Preprocessing to Dataset**
```python
texts = ["Rojan and the Lost Word", "Land of Words held magic and meaning"]
texts_tokens = [preprocessing_text(text) for text in texts]
```
**Explanation:**
Each text (sentence) is passed through our preprocessing function to generate a list of lists, where each inner list contains the cleaned tokens of one sentence.

**Example Output:**
```python
[['rojan', 'lost', 'word'],
 ['land', 'word', 'held', 'magic', 'meaning']]
```

---

### **Step 5: Train Word2Vec Model**
```python
model = Word2Vec(sentences=texts_tokens, vector_size=100, window=10, min_count=1, workers=4)
```
**Explanation:**
This line trains the **Word2Vec** model with the following parameters:
- `sentences` → tokenized data.
- `vector_size` → dimensionality of word embeddings (100 dimensions).
- `window` → context window size for neighboring words (10 means 10 words left/right).
- `min_count` → ignores words that appear less than once.
- `workers` → number of CPU cores used for training.

You can tune these parameters for better accuracy:
- Increase `vector_size` (e.g., 200 or 300) for deeper relationships.
- Increase `window` for more global context.
- Increase training corpus size for robustness.

---

### **Step 6: Extract Similar Words and Probabilities**
```python
courage_similar = model.wv.most_similar('courage', topn=10)
courage_probs = [score for w, score in courage_similar]
courage_words = [w for w, score in courage_similar]
courage_words_df = pd.DataFrame({
    'courage_words': courage_words,
    'courage_pro': courage_probs
})
courage_words_df
```
**Explanation:**
- Finds the top 10 most similar words to *courage*.
- Extracts the similarity scores.
- Creates a DataFrame containing the words and their similarity probabilities.

**Example Output:**
| courage_words | courage_pro |
|----------------|--------------|
| brave | 0.89 |
| strength | 0.85 |
| fearless | 0.82 |

---

### **Step 7: Visualize the Results**
```python
fig = px.histogram(courage_words_df, x='courage_words', y='courage_pro',
                   title='Word Similarity with Courage',
                   labels={'courage_words': 'Words', 'courage_pro': 'Similarity Probability'})
fig.show()
```
**Explanation:**
This visualization shows how semantically related each word is to *courage*, helping us understand how well the model captured relationships.

---

## 🧩 Model Improvement Tips
To make your Word2Vec model more accurate:
1. **Increase dataset size** – The more text, the better the embeddings.
2. **Tune parameters** – Try different `vector_size`, `window`, and `min_count` values.
3. **Clean data carefully** – Remove unnecessary symbols, stopwords, and rare words.
4. **Visualize embeddings** – Use tools like `TSNE` or `PCA` to visualize clusters.

---

## 🧪 Possible Extensions
- Use a larger corpus (e.g., Persian or English story datasets).
- Compare Word2Vec results with BERT or FastText models.
- Apply similarity measures to sentiment analysis.
- Create an interactive visualization dashboard.

---

## 📚 Summary
This project builds a **complete NLP pipeline** from raw text to trained Word2Vec embeddings and visualizations. It highlights how AI models can capture **semantic meaning** and relationships between words, serving as a foundation for tasks like text classification, recommendation systems, or chatbot intelligence.

---

**Author:** Arshia Saberian  
**Field:** NLP | Machine Learning | Deep Learning | Computer Vision (YOLO)  
**Purpose:** Academic Research & AI Model Experimentation

