# Sentiment Analysis on Social Media Data (Task 4)

## 📌 Project Overview
This project analyzes and visualizes sentiment patterns in social media data to understand public opinion and attitudes towards different topics or brands.

The dataset is provided by [Prodigy InfoTech](https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%204) and contains tweets with associated entities and sentiment labels.  
We use **Natural Language Processing (NLP)** techniques to clean the data, analyze sentiments, and visualize the results with both static and interactive charts.

---

## 📂 Dataset
The dataset file used:
- `twitter_training.csv`

**Columns:**
1. `ID` – Unique identifier for the tweet  
2. `Entity` – Topic or brand associated with the tweet  
3. `OriginalSentiment` – Pre-labelled sentiment (can be ignored for fresh scoring)  
4. `TweetText` – Actual tweet text  

---

## ⚙️ Tech Stack
- **Python 3**
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `plotly` (interactive charts)
  - `nltk` (VADER sentiment analysis)
  - `wordcloud`
  - `scikit-learn` (TF-IDF)

---

## 📊 Features
- **Data Cleaning**:
  - Remove URLs, mentions, hashtags, punctuation
  - Handle missing or non-string values  
- **Sentiment Scoring**:
  - Use `VADER` from `nltk` to calculate sentiment polarity  
  - Classify as `positive`, `negative`, or `neutral`
- **Visualizations**:
  - Overall sentiment distribution (static)
  - Entity-level sentiment breakdown (interactive bar chart)
  - Compound score distribution per entity (interactive box plot)
  - Word clouds for positive & negative tweets
  - TF-IDF weighted word cloud

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tanmayy06/PRODIGY_DS_04
   cd <PRODIGY_DS_04>
Install Dependencies

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn plotly nltk wordcloud scikit-learn
Download NLTK Resources

bash
Copy
Edit
python -c "import nltk; nltk.download('vader_lexicon')"
Place Dataset

Make sure twitter_training.csv is in the same directory as task_4.py.

Run the Script

bash
Copy
Edit
python task_4.py
📷 Sample Outputs
Static Charts (Matplotlib / Seaborn)

Interactive Charts (Plotly)

Word Clouds

📈 Example Insights
Which brands/topics receive the most positive or negative tweets.

Sentiment intensity distribution per entity.

Common words used in positive vs. negative tweets.
Author 
Tanmay Gupta 
Data Science Intern 
