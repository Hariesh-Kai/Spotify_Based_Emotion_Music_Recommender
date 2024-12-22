import os
import cv2
import numpy as np
import streamlit as st
import joblib
from tensorflow import keras
import mediapipe as mp
import pandas as pd
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify credentials from Streamlit secrets
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mediapipe_3emotion_model_1.h5')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'Label_Encoder', 'label_encoder_3_emotion.pkl')

@st.cache_resource
def load_resources():
    try:
        model = keras.models.load_model(MODEL_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        return model, le
    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        return None, None

model, le = load_resources()

# Extract face landmarks using MediaPipe
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp.solutions.face_mesh.FaceMesh(static_image_mode=True).process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        landmark_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark])
        return landmark_array.flatten()
    return None

# Process uploaded image for emotion prediction
def process_image(image):
    landmarks = extract_landmarks(image)
    if landmarks is not None:
        prediction = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_class = np.argmax(prediction)
        emotion = le.inverse_transform([predicted_class])[0]
        return emotion
    return None

# Recommend songs based on similarity
def recommend(song, music, similarity):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_music_names = []
    recommended_music_posters = []
    recommended_music_links = []

    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        album_cover_url, track_url = get_song_album_cover_url_and_track_url(music.iloc[i[0]].song, artist)
        recommended_music_posters.append(album_cover_url)
        recommended_music_names.append(music.iloc[i[0]].song)
        recommended_music_links.append(track_url)

    return recommended_music_names, recommended_music_posters, recommended_music_links

# Get album cover URL and Spotify track URL
def get_song_album_cover_url_and_track_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    if results and results["tracks"]["items"]:
        track_info = results["tracks"]["items"][0]
        album_cover_url = track_info["album"]["images"][0]["url"]
        track_url = track_info["external_urls"]["spotify"]
        return album_cover_url, track_url
    return "https://i.postimg.cc/0QNxYz4V/social.png", None

# Load music and similarity data dynamically
def load_music_data(emotion):
    dataframe_path = os.path.join(BASE_DIR, 'pickle', 'dataframe', f'{emotion.lower()}_df.pkl')
    similarity_path = os.path.join(BASE_DIR, 'pickle', 'similarity', f'{emotion.lower()}_similarity.pkl')
    with open(dataframe_path, 'rb') as df_file, open(similarity_path, 'rb') as sim_file:
        music = pickle.load(df_file)
        similarity = pickle.load(sim_file)
    return music, similarity
