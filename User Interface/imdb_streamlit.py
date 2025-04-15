import joblib
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained vectorizer and model
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
log_reg = joblib.load("log_reg_model.pkl")

# Function to get movie reviews from TMDB API
def get_movie_reviews(movie_name: str, api_token: str):
    base_url = 'https://api.themoviedb.org/3'
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Accept': 'application/json'
    }
    search_response = requests.get(
        f'{base_url}/search/movie',
        headers=headers,
        params={'query': movie_name}
    ).json()

    if not search_response['results']:
        return None, None, 0

    movie_id = search_response['results'][0]['id']

    # Get Reviews
    page = 1
    all_reviews = []
    while True:
        review_response = requests.get(
            f'{base_url}/movie/{movie_id}/reviews',
            headers=headers,
            params={'page': page}
        ).json()

        reviews = review_response.get('results', [])
        all_reviews.extend([r['content'] for r in reviews])

        if page >= review_response['total_pages']:
            break
        page += 1

    return all_reviews, search_response['results'][0]['title'], len(all_reviews)

# Streamlit UI
st.title("🎬 IMDb Movie Review Sentiment Analysis")

# User input for movie name
movie_name = st.text_input("Enter a movie name:", "Titanic")

# API Token
TMDB_API_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiZWQ4YjVhM2M1ODBlNzJkNjdmYjI2M2IwZjI2MGM4ZCIsIm5iZiI6MTc0MTQwNDAzOC44OTQsInN1YiI6IjY3Y2JiNzg2NDJjNzUyMTI1MmY1OTQ4MyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.qHPK540tVsfSm9OoL51lkEPRHET9PVXq0s5ej3a3C_8'

if st.button("Analyze Sentiment"):
    reviews, title, total_reviews = get_movie_reviews(movie_name, TMDB_API_TOKEN)

    if reviews is None:
        st.error(f"No movie found with name: {movie_name}")
    elif total_reviews == 0:
        st.warning("No reviews found for this movie.")
    else:
        st.success(f"Found {total_reviews} reviews for **{title}**!")

        if total_reviews > 0:
            # Transform reviews into TF-IDF features using the pre-trained vectorizer
            X_test_tfidf = tfidf_vectorizer.transform(reviews)

            # Predict sentiment using the pre-trained logistic regression model
            y_pred = log_reg.predict(X_test_tfidf)

            # Calculate sentiment percentages
            positive_count = sum(y_pred == 1)
            negative_count = sum(y_pred == 0)
            positive_percentage = (positive_count / total_reviews) * 100
            negative_percentage = (negative_count / total_reviews) * 100

            # Display sentiment results
            st.write(f"**Positive Reviews:** {positive_percentage:.2f}%")
            st.write(f"**Negative Reviews:** {negative_percentage:.2f}%")

            # Plot sentiment distribution
            fig, ax = plt.subplots()
            sns.barplot(x=["Positive", "Negative"], y=[positive_percentage, negative_percentage], palette="coolwarm", ax=ax)
            ax.set_ylabel("Percentage")
            ax.set_title(f"Sentiment Distribution for {title}")
            st.pyplot(fig)
