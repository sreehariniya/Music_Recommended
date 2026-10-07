import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Personalized Music Recommendation",
    page_icon="🎵",
    layout="centered"
)


# =========================================================
# CUSTOM CSS - BACKGROUND AND DESIGN
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(145, 100, 255, 0.22),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(230, 100, 200, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 90%,
            rgba(80, 150, 255, 0.15),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #0b0915 0%,
            #171329 50%,
            #0c1020 100%
        );

    color: white;
}


/* Page width */
.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* Music icons */
.music-icons {
    text-align: center;
    font-size: 48px;
    margin-bottom: 12px;
    letter-spacing: 12px;
}


/* Main title */
.main-title {
    text-align: center;
    color: white;
    font-size: 46px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 15px;
}


/* Subtitle */
.subtitle {
    text-align: center;
    color: #d2ccdf;
    font-size: 18px;
    margin-bottom: 40px;
}


/* Section headings */
.section-title {
    color: white;
    font-size: 24px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 18px;
}


/* Labels */
label {
    color: #eeeeee !important;
    font-size: 16px !important;
}


/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.09) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
}


/* Button */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
    background: linear-gradient(
        90deg,
        #9167e8,
        #c06bdd
    );
    color: white;
    font-size: 17px;
    font-weight: 700;
    padding: 13px;
    margin-top: 15px;
    transition: 0.3s;
}


.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0px 8px 25px
        rgba(160,100,255,0.35);
}


/* Recommendation cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    margin-top: 12px;
    margin-bottom: 12px;
    padding: 5px;
}


/* Footer */
.footer {
    text-align: center;
    color: #9992a9;
    font-size: 13px;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="music-icons">🎵 🎧 🎶</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">'
    'Personalized Music Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover songs that match your music taste '
    'using Machine Learning ✨'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

try:

    df = pd.read_csv("music_data.csv")

except FileNotFoundError:

    st.error(
        "❌ music_data.csv was not found. "
        "Please upload it to the same folder as app.py."
    )

    st.stop()


# Clean column names
df.columns = df.columns.str.strip().str.lower()


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "song",
    "artist",
    "genre",
    "energy",
    "danceability",
    "acousticness",
    "valence"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Missing columns in music_data.csv: "
        + ", ".join(missing_columns)
    )

    st.stop()


# =========================================================
# MACHINE LEARNING FEATURES
# =========================================================

features = [
    "energy",
    "danceability",
    "acousticness",
    "valence"
]


# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

feature_matrix = scaler.fit_transform(
    df[features]
)


# =========================================================
# COSINE SIMILARITY
# =========================================================

similarity_matrix = cosine_similarity(
    feature_matrix
)


# =========================================================
# USER INPUT
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🎧 Choose Your Music'
    '</div>',
    unsafe_allow_html=True
)


# Song selection
selected_song = st.selectbox(
    "Select a song",
    df["song"].tolist()
)


# Number of recommendations
number_of_recommendations = st.selectbox(
    "How many songs would you like?",
    [1, 2, 3, 4, 5],
    index=2
)


# =========================================================
# RECOMMEND BUTTON
# =========================================================

if st.button("🎵 Recommend Songs"):

    # Find selected song
    song_index = df[
        df["song"] == selected_song
    ].index[0]


    # Get similarity scores
    similarity_scores = list(
        enumerate(
            similarity_matrix[song_index]
        )
    )


    # Sort by similarity
    similarity_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # =====================================================
    # RECOMMENDATION TITLE
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '✨ Recommended For You'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # DISPLAY RECOMMENDATIONS
    # =====================================================

    count = 0


    for index, score in similarity_scores:

        # Don't recommend the selected song
        if index == song_index:
            continue


        # Get song information
        song = df.iloc[index]["song"]
        artist = df.iloc[index]["artist"]
        genre = df.iloc[index]["genre"]


        # -----------------------------------------------
        # Recommendation Card
        # -----------------------------------------------

        with st.container(border=True):

            st.markdown(
                f"### 🎵 {song}"
            )

            st.write(
                f"**{artist}** • {genre}"
            )

            st.caption(
                f"✨ Similarity Score: {score:.2f}"
            )


        count += 1


        if count == number_of_recommendations:
            break


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '🎶 Powered by Python • Machine Learning • Streamlit'
    '</div>',
    unsafe_allow_html=True
)
