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
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify credentials from Streamlit secrets
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load model and label encoder dynamically

def load_model_and_encoder():
    model = keras.models.load_model('./models/mediapipe_3emotion_model_1.h5')
    model_path = os.path.join(os.getcwd(), 'Label_Encoder', 'label_encoder_3_emotion.pkl')
    le = joblib.load(model_path)
    return model, le

model, le = load_model_and_encoder()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Extract face landmarks using MediaPipe
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

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
@st.cache_data
def load_music_data(emotion):
    with open(f'./pickle/dataframe/{emotion.lower()}_df.pkl', 'rb') as f:
        music = pickle.load(f)
    with open(f'./pickle/similarity/{emotion.lower()}_similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    return music, similarity

# Streamlit app interface
st.title("Emotion-Based Music Recommender")

# Step 1: Upload image for emotion detection
uploaded_file = st.file_uploader("Upload an image with a face", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Step 2: Detect emotion
    emotion = process_image(image)
    if emotion:
        st.write(f"Detected Emotion: {emotion}")

        # Step 3: Load data for recommendations
        music, similarity = load_music_data(emotion)

        # Step 4: Select song and show recommendations
        song_list = music['song'].values
        selected_song = st.selectbox("Type or select a song", song_list)

        if st.button('Show Recommendations'):
            recommended_music_names, recommended_music_posters, recommended_music_links = recommend(selected_song, music, similarity)
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(recommended_music_names):
                    with col:
                        st.text(recommended_music_names[i])
                        if recommended_music_links[i]:
                            link = f'<a href="{recommended_music_links[i]}" target="_blank"><img src="{recommended_music_posters[i]}" width="100" /></a>'
                            st.markdown(link, unsafe_allow_html=True)
    else:
        st.error("No face landmarks detected. Please try another image.")
