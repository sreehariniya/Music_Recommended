
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Personalized Music Recommendation System")

st.write(
    "Select a song and get personalized song recommendations "
    "based on music similarity."
)

# Load dataset
df = pd.read_csv("music_data.csv")

# Features
features = [
    "energy",
    "danceability",
    "acousticness",
    "valence"
]

# Scale features
scaler = StandardScaler()
feature_matrix = scaler.fit_transform(df[features])

# Calculate similarity
similarity_matrix = cosine_similarity(feature_matrix)

# Select song
song_list = df["song"].tolist()

selected_song = st.selectbox(
    "🎧 Select a song",
    song_list
)

number_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=5,
    value=3
)

if st.button("🎵 Recommend Songs"):

    song_index = df[
        df["song"] == selected_song
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[song_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("🎶 Recommended Songs")

    count = 0

    for index, score in similarity_scores:

        if index == song_index:
            continue

        song = df.iloc[index]["song"]
        artist = df.iloc[index]["artist"]
        genre = df.iloc[index]["genre"]

        st.write(
            f"🎵 **{song}** - {artist}  \n"
            f"Genre: {genre} | "
            f"Similarity: {score:.2f}"
        )

        count += 1

        if count == number_of_recommendations:
            break
