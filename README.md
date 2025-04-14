# Intelligent Sentiment Analysis System for Movie Reviews
## 📌 Project Purpose
- Intelligent sentiment analysis system that can classify movie reviews as either positive or negative.
- Leverage **NLP** and **Machine Learning** to develop a model that can automatically determine audience sentiment based on movie reviews.
- Relevance & Impact: Huge Volume of Reviews, Decision-Making Tool, Automation through NLP

## 📊 Dataset Description
We used a **Kaggle dataset** and **TMDb API** for analysis:

- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) – Contains 50,000 labeled IMDb movie reviews from Kaggle used for model training and evaluation.

- [TMDb API](https://developer.themoviedb.org/docs/getting-started) – Used to fetch real-time movie reviews via HTTP GET requests for sentiment analysis within the user interface.

📖 Data Dictionary (`IMDB Dataset 2.csv`)
 | Column Name      | Description                                      |
 |------------------|--------------------------------------------------|
 | `review`         | Full text of the movie review                    |
 | `sentiment`      | Sentiment label: positive or negative            |


## 💡 Methodology
1️⃣ **Data Preprocessing**
- Standardized and cleaned text data for consistency
- Removed duplicates and handled missing values to ensure data quality
- Prepared sentiment labels and text for machine learning analysis
  
*Final dataset saved as `Cleaned_IMDB_dataset.csv`*

2️⃣ **Feature Selection**
- Used **TF-IDF** and **Word Embeddings** to convert review text into machine-readable vectors

3️⃣ **Model Training**
- Built three models using TF-IDF features:
  - Logistic Regression
  - Support Vector Machine
  - Random Forest
- Identified **Logistic Regression** as the top performer, then re-trained it using Word Embeddings for comparison

4️⃣ **Model Evaluation**
- Assessed model performance using **confusion matrix** and **ROC curve**
- Used an **open-source LLM** to cross-validate predictions from both feature extraction methods

5️⃣ **Web Scraping**
- Integrated with TMDb API to fetch real-time movie reviews by movie title
- Parsed paginated responses, handled errors, and structured reviews into a DataFrame for use in the interface

6️⃣ **User Interface Deployment**
- Developed an interactive UI using **Streamlit** for real-time sentiment analysis
- Connected to the TMDb API to fetch reviews based on user-input movie titles
- Loaded a pre-trained **TF-IDF + Logistic Regression** model to classify review sentiment
- Displayed results as positive vs. negative percentages with a simple bar chart

## 🔍 Key Findings
- ✅ TF-IDF + Logistic Regression achieved the highest accuracy (88.78%), with an AUC of 0.96
- SVM and Random Forest followed at 88.28% and 83.49% respectively
- Word Embeddings + LR underperformed, reaching only 75.73% accuracy with an AUC of 0.83
- 🧠 LLM-based validation revealed that:
  - TF-IDF + LR retained strong generalization: 84% avg. precision, 0.79 AUC
  - Word Embeddings + LR dropped significantly: 56% avg. precision, 0.64 AUC

📌 Final model selection: **TF-IDF + Logistic Regression**, based on its consistency and superior performance across both traditional evaluation and LLM validation.

## 🚀 Business Recommendations
• Use the model for real-time feedback on movie releases and trailer sentiment.
• Extend the system to multilingual reviews to serve a global audience.• Leverage keyword frequency from positive/negative reviews to improve marketing messaging.

## 📌 Conclusion
The project successfully built a high-performing sentiment classifier using TF-IDF and Logistic Regression. With real-time OMDb integration and a user-friendly Streamlit UI, this tool automates review analysis, offering practical value to media, film, and marketing professionals.

## 🛠 Technologies Used
• Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
• NLP Tools (NLTK, WordCloud)
• Streamlit (for UI deployment)
• OMDb API (for real-time movie review retrieval)

## 📂 Repository Files

• IMDB Dataset 2.csv – IMDb movie reviews with sentiment labels• sentiment_analysis.ipynb – Jupyter notebook for training and evaluation• app.py – Streamlit interface for live predictions• logistic_model.pkl – Saved Logistic Regression model• tfidf_vectorizer.pkl – Saved TF-IDF vectorizer• README.md – Project documentation
