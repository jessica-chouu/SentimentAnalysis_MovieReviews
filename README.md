# 🎬 Intelligent Sentiment Analysis System for Movie Reviews
## 🎯 Project Purpose
- Intelligent sentiment analysis system that can classify movie reviews as either positive or negative.
- Leverage **NLP** and **Machine Learning** to develop a model that can automatically determine audience sentiment based on movie reviews.
<br>

## 📊 Dataset Description
We used a **Kaggle dataset** and **TMDb API** for analysis:

- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) – Contains 50,000 labeled IMDb movie reviews from Kaggle used for model training and evaluation.

- [TMDb API](https://developer.themoviedb.org/docs/getting-started) – Used to fetch real-time movie reviews via HTTP GET requests for sentiment analysis within the user interface.

📖 Data Dictionary (`IMDB Dataset 2.csv`)
 | Column Name      | Description                                      |
 |------------------|--------------------------------------------------|
 | `review`         | Full text of the movie review                    |
 | `sentiment`      | Sentiment label: positive or negative            |

<br>

## 💡 Methodology
1️⃣ **Data Preprocessing**
- Standardized and cleaned text data for consistency.
- Removed duplicates and handled missing values to ensure data quality.
- Prepared sentiment labels and text for machine learning analysis.
  
📖 Cleaned Dataset:
   
   <img src="Data%20Visualization/1%20-%20cleaned%20dataset.png" width="500"/>

2️⃣ **Feature Selection**
- Used **TF-IDF** and **Word Embeddings** to convert review text into machine-readable vectors.

3️⃣ **Model Training**
- Built three models using TF-IDF features:
  - Logistic Regression
  - Support Vector Machine
  - Random Forest
- Identified **Logistic Regression** as the top performer, then re-trained it using Word Embeddings for comparison.

4️⃣ **Model Evaluation**
- Assessed model performance using **confusion matrix** and **ROC curve**.
- Used an **open-source LLM** to cross-validate predictions from both feature extraction methods.

5️⃣ **Web Scraping**
- Integrated with TMDb API to fetch real-time movie reviews by movie title.
- Parsed paginated responses, handled errors, and structured reviews into a DataFrame for use in the interface.

6️⃣ **User Interface Deployment**
- Developed an interactive UI using **Streamlit** for real-time sentiment analysis.
- Connected to the TMDb API to fetch reviews based on user-input movie titles.
- Loaded a pre-trained **TF-IDF + Logistic Regression** model to classify review sentiment.
- Displayed results as positive vs. negative percentages with a simple bar chart.

<br>

## 🔍 Key Findings
- **TF-IDF + Logistic Regression achieved the highest accuracy (88.78%)**, followed by SVM (88.28%) and Random Forest (83.49%).
<img src="Data%20Visualization/2%20-%20Camparison%20of%20model.png" width="500"/>

- **TF-IDF outperformed Word Embeddings**, with Logistic Regression scoring 88.78% vs. 75.73%, showing better feature representation.
<img src="Data%20Visualization/3%20-%20Comparison%20of%20TF%20and%20WE.png" width="500"/>

- **TF-IDF + LR** showed balanced predictions with **a high AUC of 0.96**, while Word Embeddings + LR had more misclassifications and a lower AUC of 0.83.
<p float="left">
  <img src="Data%20Visualization/4%20-%20CM%20(TF).png" width="45%" />
  <img src="Data%20Visualization/4%20-%20ROC%20(TF).png" width="45%" />
</p>

- **LLM validation** confirmed that **TF-IDF features generalized better (AUC = 0.86)** vs. Word Embeddings (AUC = 0.52).
<p float="left">
  <img src="Data%20Visualization/6%20-%20LLM%20CM(TF).png" width="45%" />
  <img src="Data%20Visualization/6%20-%20LLM%20ROC(TF).png" width="45%" />
</p>

📍 Final model selection: **TF-IDF + Logistic Regression**, based on its consistency and superior performance across both traditional evaluation and LLM validation.

<br>

## 🖥️ User Interface Demo
 Try the app: [🎬 IMDb Movie Review Sentiment Analysis](https://imdb-sentiment-analysis-movie-reviews.streamlit.app/)
<div align="center">
   <img src="Data%20Visualization/8%20-%20Streamlit%201.jpg" width="60%" /> 
   <img src="Data%20Visualization/8%20-%20Streamlit%202.jpg" width="60%" />
</div>

 How It Works:
1. **User Input**: Users enter a movie name in the input field, and the system automatically retrieves relevant reviews using the TMDb API.
2. **Sentiment Analysis**: The retrieved reviews are processed using a TF-IDF vectorizer(`tfidf_vectorizer.pkl`) and Logistic Regression model(`log_reg_model.pkl`) to classify them as positive or negative.
3. **Results Display**: The system presents the number of positive and negative reviews, along with a bar chart visualization of the sentiment distribution.

<br>

## 📌 Conclusion
The project built a high-performing sentiment classifier using TF-IDF and Logistic Regression. With real-time TMDb integration and a user-friendly Streamlit UI, this tool automates review analysis, offering practical value to media, film, and marketing professionals.

<br>

## 🚀 Strategic Opportunities
- Apply sentiment insights to optimize promotional timing, messaging, and competitor benchmarking.
- Extend functionality by integrating topic modeling to extract trending themes from reviews.
- Explore multilingual sentiment classification to expand market coverage and global usability.

<br>

## 🛠 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib)
- **NLP** (NLTK)
- **Machine Learning** (Logistic Regression, SVM, Random Forest)
- **Visualization** (Seaborn, Matplotlib)
- **UI Deployment** (Streamlit)
- **APIs** (TMDb API, Together AI API)

<br>

## 📂 Repository Files
- `IMDB Dataset.csv.gz` — IMDb movie reviews with sentiment labels  
- `imdb_sentiment_analysis.ipynb` — Main notebook for data preprocessing, model training, and model evaluation  
- `imdb_streamlit.py` — Streamlit script to deploy the sentiment analysis web app  
- `log_reg_model.pkl` — Pre-trained Logistic Regression model for classifying sentiment  
- `tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer used to transform review text for prediction  
- `requirements.txt` — List of required Python packages for running the Streamlit app  
- `Data Visualization/` — Folder containing visualizations (confusion matrices, ROC curves, comparison charts) used in the analysis  
- `README.md` — Project documentation
