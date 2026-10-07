import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Personalized Music Recommendation",
    page_icon="🎵",
    layout="centered"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(177, 137, 255, 0.20), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(255, 120, 190, 0.18), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(100, 180, 255, 0.15), transparent 30%),
        linear-gradient(135deg, #0b0b14 0%, #171327 50%, #0d1020 100%);
    color: white;
}

/* Main content */
.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cfc9df;
    margin-bottom: 35px;
}

/* Music decoration */
.music-decoration {
    text-align: center;
    font-size: 42px;
    margin-bottom: 5px;
}

/* Glass card */
.music-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    padding: 28px;
    margin-top: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.30);
    backdrop-filter: blur(12px);
}

/* Section heading */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 15px;
}

/* Recommendation card */
.recommendation {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px 20px;
    margin: 12px 0;
    border-left: 4px solid #b388ff;
}

/* Song name */
.song-name {
    font-size: 19px;
    font-weight: 700;
    color: white;
}

/* Artist */
.artist-name {
    font-size: 15px;
    color: #c9c1d9;
}

/* Similarity */
.similarity {
    font-size: 14px;
    color: #d7bfff;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 13px;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(90deg, #9c6cff, #d06cff);
    color: white;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 25px rgba(180,100,255,0.35);
}

/* Select boxes */
div[data-baseweb="select"] > div {
    border-radius: 14px;
    background-color: rgba(255,255,255,0.08);
}

/* Footer */
.footer {
    text-align: center;
    color: #aaa2b8;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">🎧 Choose Your Music</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover songs that match your music taste using Machine Learning ✨'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("music_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()


# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

features = [
    "energy",
    "danceability",
    "acousticness",
    "valence"
]


# ---------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------

scaler = StandardScaler()

feature_matrix = scaler.fit_transform(
    df[features]
)

similarity_matrix = cosine_similarity(
    feature_matrix
)


# ---------------------------------------------------
# USER INPUT CARD
# ---------------------------------------------------

# ---------------------------------------------------
# USER INPUT CARD
# ---------------------------------------------------

st.markdown(
    '<div class="music-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🎧 Choose Your Music</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🎧 Choose Your Music</div>',
    unsafe_allow_html=True
)

# Song selection
selected_song = st.selectbox(
    "Select a song",
    df["song"].tolist()
)

# Number selection using dropdown
number_of_recommendations = st.selectbox(
    "How many songs would you like?",
    [1, 2, 3, 4, 5],
    index=2
)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# RECOMMENDATION BUTTON
# ---------------------------------------------------

st.write("")

if st.button("🎵 Recommend Songs"):

    song_index = df[
        df["song"] == selected_song
    ].index[0]

    similarity_scores = list(
        enumerate(
            similarity_matrix[song_index]
        )
    )

    similarity_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    st.markdown(
    '<div class="section-title">✨ Recommended For You</div>',
    unsafe_allow_html=True
    )

    count = 0

    for index, score in similarity_scores:

        # Don't recommend the selected song
        if index == song_index:
            continue

        song = df.iloc[index]["song"]
        artist = df.iloc[index]["artist"]
        genre = df.iloc[index]["genre"]

        st.markdown(
            f"""
            <div class="recommendation">
                <div class="song-name">🎵 {song}</div>
                <div class="artist-name">
                    {artist} • {genre}
                </div>
                <div class="similarity">
                    ✨ Similarity Score: {score:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        count += 1

        if count == number_of_recommendations:
            break

   


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
    '<div class="footer">'
    '🎶 Powered by Python • Machine Learning • Streamlit'
    '</div>',
    unsafe_allow_html=True
)
