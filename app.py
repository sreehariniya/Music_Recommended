import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Personalized Music Recommendation",
    page_icon="🎵",
    layout="centered"
)


# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

/* ---------- BACKGROUND ---------- */

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


/* ---------- PAGE WIDTH ---------- */

.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* ---------- MUSIC ICONS ---------- */

.music-icons {
    text-align: center;
    font-size: 50px;
    margin-bottom: 10px;
    letter-spacing: 15px;
}


/* ---------- MAIN TITLE ---------- */

.main-title {
    text-align: center;
    color: white;
    font-size: 48px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 15px;
}


/* ---------- SUBTITLE ---------- */

.subtitle {
    text-align: center;
    color: #d2ccdf;
    font-size: 18px;
    margin-bottom: 45px;
}


/* ---------- SECTION TITLE ---------- */

.section-title {
    color: white;
    font-size: 24px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* ---------- LABELS ---------- */

label {
    color: #eeeeee !important;
    font-size: 16px !important;
}


/* ---------- SELECT BOX ---------- */

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.09) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
    color: white !important;
}


/* ---------- BUTTON ---------- */

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


/* ---------- RECOMMENDATION CARD ---------- */

.recommendation-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 17px 20px;
    margin-top: 12px;
    box-shadow:
        0px 8px 25px rgba(0,0,0,0.18);
}


.song-title {
    color: white;
    font-size: 19px;
    font-weight: 700;
}


.song-info {
    color: #c9c1d8;
    font-size: 15px;
    margin-top: 4px;
}


.similarity {
    color: #c59aff;
    font-size: 14px;
    margin-top: 7px;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #9992a9;
    font-size: 13px;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

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


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("music_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()


# =====================================================
# FEATURES
# =====================================================

features = [
    "energy",
    "danceability",
    "acousticness",
    "valence"
]


# =====================================================
# MACHINE LEARNING
# =====================================================

scaler = StandardScaler()

feature_matrix = scaler.fit_transform(
    df[features]
)

similarity_matrix = cosine_similarity(
    feature_matrix
)


# =====================================================
# USER INPUT
# =====================================================

st.markdown(
    '<div class="section-title">🎧 Choose Your Music</div>',
    unsafe_allow_html=True
)


selected_song = st.selectbox(
    "Select a song",
    df["song"].tolist()
)


number_of_recommendations = st.selectbox(
    "How many songs would you like?",
    [1, 2, 3, 4, 5],
    index=2
)


# =====================================================
# BUTTON
# =====================================================

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


    # ================================================
    # RECOMMENDATION TITLE
    # ================================================

    st.markdown(
        '<div class="section-title">'
        '✨ Recommended For You'
        '</div>',
        unsafe_allow_html=True
    )


    # ================================================
    # DISPLAY RECOMMENDATIONS
    # ================================================

    count = 0

    for index, score in similarity_scores:

        # Don't recommend selected song
        if index == song_index:
            continue


        song = df.iloc[index]["song"]
        artist = df.iloc[index]["artist"]
        genre = df.iloc[index]["genre"]


        st.markdown(
            f"""
            <div class="recommendation-card">

                <div class="song-title">
                    🎵 {song}
                </div>

                <div class="song-info">
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


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    '<div class="footer">'
    '🎶 Powered by Python • Machine Learning • Streamlit'
    '</div>',
    unsafe_allow_html=True
)
