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

# Load the trained model and label encoder
MODEL_PATH = 'D:/Music_Recommender/models/mediapipe_3emotion_model_1.h5'
LABEL_ENCODER_PATH = 'D:/Music_Recommender/Label_Encoder/label_encoder_3_emotion.pkl'
model = keras.models.load_model(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Spotify credentials
CLIENT_ID = "defbade4e685489eb7339f1c0d2e7817"
CLIENT_SECRET = "8ff0954c9a21473f99f164a55aa4de0b"
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Get album cover URL and audio preview URL
def get_song_album_cover_url_and_preview_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    if results and results["tracks"]["items"]:
        track_info = results["tracks"]["items"][0]
        album_cover_url = track_info["album"]["images"][0]["url"]
        preview_url = track_info["preview_url"]  # Get the 30-second preview URL
        track_id = track_info["id"]  # Get the Spotify track ID
        return album_cover_url, preview_url, track_id
    return "https://i.postimg.cc/0QNxYz4V/social.png", None, None

# Recommend songs based on similarity
def recommend(song, music, similarity):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_music_names = []
    recommended_music_posters = []
    recommended_music_previews = []
    recommended_music_ids = []

    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        album_cover_url, preview_url, track_id = get_song_album_cover_url_and_preview_url(music.iloc[i[0]].song, artist)
        recommended_music_posters.append(album_cover_url)
        recommended_music_names.append(music.iloc[i[0]].song)
        recommended_music_previews.append(preview_url)
        recommended_music_ids.append(track_id)

    return recommended_music_names, recommended_music_posters, recommended_music_previews, recommended_music_ids

# Extract face landmarks using MediaPipe
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        landmark_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark])
        return landmark_array.flatten()
    return None

# Process each video frame to predict emotion
def process_frame(frame):
    landmarks = extract_landmarks(frame)
    if landmarks is not None:
        prediction = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_class = np.argmax(prediction)
        emotion = le.inverse_transform([predicted_class])[0]
        return frame, emotion
    return frame, None

# Real-time emotion detection using webcam
def real_time_emotion_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    st.write("Detecting emotion...")

    video_placeholder = st.empty()
    latest_emotion = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        frame, detected_emotion = process_frame(frame)
        video_placeholder.image(frame, channels="BGR")

        if detected_emotion:
            st.session_state['detected_emotion'] = detected_emotion
            st.write(f"Detected Emotion: {detected_emotion}")
            cap.release()  # Release webcam after detecting emotion
            break

        time.sleep(0.1)

    cap.release()

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

# Step 1: Start emotion detection
if st.button("Start Emotion Detection"):
    real_time_emotion_detection()

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
    recommended_music_names, recommended_music_posters, recommended_music_previews, recommended_music_ids = recommend(selected_song, music, similarity)

    # Display recommendations in a two-column layout
    cols = st.columns(2)  # Create two columns

    for i in range(len(recommended_music_names)):
        col = cols[i % 2]  # Alternate between the two columns (left and right)

        with col:
            st.markdown(
                f"""
                <div style="
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    margin-bottom: 20px;">
                    <iframe src="https://open.spotify.com/embed/track/{recommended_music_ids[i]}" 
                            width="100%" height="80" frameborder="0" 
                            allowtransparency="true" allow="encrypted-media" 
                            style="border-radius: 12px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                    </iframe>
                </div>
                """,
                unsafe_allow_html=True
            )