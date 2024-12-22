import os
import cv2
import numpy as np
import streamlit as st
import joblib
from tensorflow import keras
import mediapipe as mp
import pandas as pd
import time
import pickle
import requests
from io import BytesIO
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the trained model and label encoder from local files
MODEL_PATH = "./models/mediapipe_3emotion_model_1.h5"
LABEL_ENCODER_PATH = "./Label_Encoder/label_encoder_3_emotion.pkl"

model = keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as file:
    le = pickle.load(file)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Spotify credentials
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

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

# Extract face landmarks using MediaPipe
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        landmark_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark])
        return landmark_array.flatten()
    return None

# Process each image to predict emotion
def process_image(image):
    landmarks = extract_landmarks(image)
    if landmarks is not None:
        prediction = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_class = np.argmax(prediction)
        emotion = le.inverse_transform([predicted_class])[0]
        return image, emotion
    return image, None

# Load music and similarity data based on detected emotion
def load_music_data(emotion):
    music = pickle.load(open(f'D:/Music_Recommender/pickle/dataframe/{emotion.lower()}_df.pkl', 'rb'))
    similarity = pickle.load(open(f'D:/Music_Recommender/pickle/similarity/{emotion.lower()}_similarity.pkl', 'rb'))
    return music, similarity

# Streamlit app setup
st.title("Real-Time Emotion-Based Music Recommender")
st.write("This app detects your emotion and recommends music based on the detected emotion.")

# Initialize session state for emotion
if 'detected_emotion' not in st.session_state:
    st.session_state['detected_emotion'] = None

# Step 1: Start emotion detection with camera input
image_file = st.camera_input("Take a photo")
if image_file:
    # Convert the uploaded image to a format usable by OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame, detected_emotion = process_image(frame)
    if detected_emotion:
        st.session_state['detected_emotion'] = detected_emotion
        st.write(f"Detected Emotion: {detected_emotion}")

# Step 2: Once emotion is detected, show recommendations
if st.session_state['detected_emotion']:
    emotion = st.session_state['detected_emotion']
    st.write(f"Emotion Detected: {emotion}")

    # Load the appropriate music dataset based on emotion
    music, similarity = load_music_data(emotion)

    # Step 3: Select song from the detected emotion's song list
    song_list = music['song'].values
    selected_song = st.selectbox("Type or select a song from the dropdown", song_list)

    # Step 4: Show recommendations based on selected song
    if st.button('Show Recommendation'):
        recommended_music_names, recommended_music_posters, recommended_music_links = recommend(selected_song, music, similarity)
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(recommended_music_names):
                with col:
                    st.text(recommended_music_names[i])
                    if recommended_music_links[i]:
                        # Display image as a clickable link
                        link = f'<a href="{recommended_music_links[i]}" target="_blank"><img src="{recommended_music_posters[i]}" width="100" /></a>'
                        st.markdown(link, unsafe_allow_html=True)
