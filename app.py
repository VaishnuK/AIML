# app.py

import streamlit as st
from recommender import recommend, get_all_titles

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Movie Recommender System")
st.markdown("Get movie recommendations based on your favorite movie!")

movie_list = get_all_titles()
selected_movie = st.selectbox("Choose a movie to get similar recommendations:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader(f"Movies similar to *{selected_movie}*:")
        for i, movie in enumerate(recommendations, start=1):
            st.markdown(f"**{i}.** {movie}")
    else:
        st.error("Movie not found. Try another title.")
