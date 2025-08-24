# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Load + Clean Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IMDbRatings_IndianMovies.csv")
    return df

df = load_data()
# Genre features
genre_dummies = df['Genre'].fillna('').str.get_dummies(sep=',')
genre_matrix = genre_dummies.values

# Talent features (TF-IDF)
df[['Director','Actor 1','Actor 2','Actor 3']] = df[['Director','Actor 1','Actor 2','Actor 3']].fillna('')
df['Talent'] = df[['Director','Actor 1','Actor 2','Actor 3']].agg(' '.join, axis=1)
tfidf = TfidfVectorizer(stop_words='english')
talent_matrix = tfidf.fit_transform(df['Talent'])

# Numeric features (Year, Duration, Rating)
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[['Year','Duration','Rating']])

# ----------------------------
# Recommendation Function
# ----------------------------
def get_recommendations(movie_name, user_rating, top_n=5):
    if movie_name not in df['Name'].values:
        return None
    
    idx = df[df['Name'] == movie_name].index[0]

    # Compute similarities
    genre_vector = genre_matrix[idx].reshape(1, -1)
    genre_scores = cosine_similarity(genre_vector, genre_matrix).flatten()

    talent_vector = talent_matrix[idx]
    talent_scores = cosine_similarity(talent_vector, talent_matrix).flatten()

    num_vector = num_features[idx].reshape(1, -1)
    num_scores = cosine_similarity(num_vector, num_features).flatten()

    # Adjust weights based on how much user liked the movie
    rt = 0 #User Rating effect modifier
    if user_rating >= 7:
        w_genre, w_talent, w_num = 0.5, 0.4, 0.1
        rt = rt+2 if user_rating<=8 else rt
    elif user_rating <= 4:
        w_genre, w_talent, w_num = 0.2, -0.6, 0.4  # penalize similar talent for disliked movies
        rt = rt+5
    else:
        w_genre, w_talent, w_num = 0.3, 0.3, 0.4
        rt = rt+3

    # Base similarity
    sim_scores = (w_genre*genre_scores + w_talent*talent_scores + w_num*num_scores)

    # IMDb rating boost
    rating_boost = df['Rating'].values / 10  # normalize to 0-1
    sim_scores = 0.7*sim_scores + rt/10*rating_boost
    sim_scores = np.maximum(sim_scores, 0)

    # Top recommendations
    rec_indices = np.argsort(sim_scores)[::-1][1:top_n+1]

    # Build output with explanations
    recommendations = []
    for rec_idx in rec_indices:
        rec = {
            "Name": df.iloc[rec_idx]['Name'],
            "Year": int(df.iloc[rec_idx]['Year']),
            "Genre": df.iloc[rec_idx]['Genre'],
            "Director": df.iloc[rec_idx]['Director'],
            "Actor 1": df.iloc[rec_idx]['Actor 1'],
            "IMDb Rating": df.iloc[rec_idx]['Rating'],
            "Score": sim_scores[rec_idx],
            "Why": []
        }

        # Explanation rules
        if len(set(df.iloc[idx]['Genre'].split(',')) & set(df.iloc[rec_idx]['Genre'].split(','))) > 0:
            rec["Why"].append("Shares similar genre(s)")
        if any(actor in df.iloc[rec_idx]['Talent'] for actor in df.iloc[idx][['Actor 1','Actor 2','Actor 3']].values if actor):
            rec["Why"].append("Common actor/actress")
        if df.iloc[idx]['Director'] == df.iloc[rec_idx]['Director'] and df.iloc[idx]['Director'] != '':
            rec["Why"].append("Same director")
        if df.iloc[rec_idx]['Rating'] >= 7.5:
            rec["Why"].append(f"High IMDb rating ({df.iloc[rec_idx]['Rating']})")

        recommendations.append(rec)

    return pd.DataFrame(recommendations)


# ----------------------------
# Example Usage
# ----------------------------
movie_choice = "Gully Boy"
user_rating = 8

recs = get_recommendations(movie_choice, user_rating)

print(f"Because you rated '{movie_choice}' ({user_rating}/10), you may also like:\n")
recs# ----------------------------
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
        st.markdown(recommendation)
        for idx, row in recommendations.iterrows():
            st.markdown(f"**{idx+1}. {row['Name']}** ({int(row['Year'])})  \n"
                        f"🎭 Genre: {row['Genre']}  \n"
                        f"🎬 Director: {row['Director']}  \n"
                        f"⭐ Score: {row['Score']:.2f}")
