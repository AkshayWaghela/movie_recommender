# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IMDbRatings_IndianMovies.csv")
    return df

df_movies = load_data()

# ----------------------------
# Preprocessing
# ----------------------------
@st.cache_data
def preprocess(df):
    df = df.copy()

    # Fill missing values
    df['Year'] = df['Year'].fillna(df['Year'].median())
    df['Duration'] = df['Duration'].fillna(100)
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())

    # Genre similarity (one-hot cosine)
    genre_dummies = df['Genre'].fillna('').str.get_dummies(sep=',')
    genre_sim = cosine_similarity(genre_dummies)

    # Talent similarity (TF-IDF)
    df[['Director','Actor 1','Actor 2','Actor 3']] = df[['Director','Actor 1','Actor 2','Actor 3']].fillna('')
    df['Talent'] = df[['Director','Actor 1','Actor 2','Actor 3']].agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    talent_matrix = tfidf.fit_transform(df['Talent'])
    talent_sim = cosine_similarity(talent_matrix)

    # Numeric similarity (Year, Duration, Rating)
    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df[['Year','Duration','Rating']])
    numeric_sim = cosine_similarity(num_features)

    return df, genre_sim, talent_sim, numeric_sim

df, genre_sim, talent_sim, numeric_sim = preprocess(df_movies)

# ----------------------------
# Recommendation Function
# ----------------------------
def get_recommendations(movie_name, user_rating):
    if movie_name not in df['Name'].values:
        return None
    
    idx = df[df['Name'] == movie_name].index[0]

    # Adjust weights based on user rating
    if user_rating >= 7:
        w_genre, w_talent, w_num = 0.5, 0.4, 0.1
    elif user_rating <= 4:
        w_genre, w_talent, w_num = 0.2, -0.6, 0.4  # dissimilar talent when disliked
    else:
        w_genre, w_talent, w_num = 0.3, 0.3, 0.4

    # Compute similarity scores
    sim_scores = (
        w_genre  * genre_sim[idx] +
        w_talent * talent_sim[idx] +
        w_num    * numeric_sim[idx]
    )
    sim_scores = np.maximum(sim_scores, 0)  # avoid negatives

    # Get top 5
    rec_indices = np.argsort(sim_scores)[::-1][1:6]
    recs = df.iloc[rec_indices][['Name','Year','Genre','Director','Actor 1']].copy()
    recs['Score'] = sim_scores[rec_indices]
    return recs.reset_index(drop=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Indian Movies Recommender (IMDb Dataset)")
st.write("Get personalized movie recommendations based on the **last movie you watched** and how much you liked it.")

# Movie dropdown
movie_choice = st.selectbox("Choose a movie you watched:", sorted(df['Name'].dropna().unique()))

# Rating slider
user_rating = st.slider("How much did you like this movie?", 1, 10, 7)

# Button
if st.button("Recommend Movies 🎥"):
    recommendations = get_recommendations(movie_choice, user_rating)

    if recommendations is None:
        st.warning("Movie not found in dataset. Try another one!")
    else:
        st.subheader(f"Because you rated **{movie_choice}** ({user_rating}/10), you may also like:")
        for idx, row in recommendations.iterrows():
            st.markdown(f"**{idx+1}. {row['Name']}** ({int(row['Year'])})  \n"
                        f"🎭 Genre: {row['Genre']}  \n"
                        f"🎬 Director: {row['Director']}  \n"
                        f"⭐ Score: {row['Score']:.2f}")
