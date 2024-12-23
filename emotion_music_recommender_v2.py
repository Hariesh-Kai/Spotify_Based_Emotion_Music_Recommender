import os
import cv2
import numpy as np
import streamlit as st
import joblib
from tensorflow import keras
import mediapipe as mp
import pickle
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Paths to model and label encoder
MODEL_PATH = './models/mediapipe_3emotion_model_1.h5'
LABEL_ENCODER_PATH = './Label_Encoder/label_encoder_3_emotion.pkl'

# Load the trained model and label encoder
model = keras.models.load_model(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Spotify API setup
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# MediaPipe Face Mesh initialization
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

# Real-time emotion detection transformer for streamlit-webrtc
class EmotionDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.le = le
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        landmarks = extract_landmarks(img)
        if landmarks is not None:
            prediction = self.model.predict(np.expand_dims(landmarks, axis=0))
            predicted_class = np.argmax(prediction)
            emotion = self.le.inverse_transform([predicted_class])[0]
            cv2.putText(img, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Spotify integration for album cover and previews
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

# Load music data based on emotion
def load_music_data(emotion):
    music = pickle.load(open(f'./pickle/dataframe/{emotion.lower()}_df.pkl', 'rb'))
    similarity = pickle.load(open(f'./pickle/similarity/{emotion.lower()}_similarity.pkl', 'rb'))
    return music, similarity

# Streamlit app
st.title("Real-Time Emotion-Based Music Recommender")
st.write("This app detects your emotion and recommends music based on the detected emotion.")

# Webcam emotion detection
st.header("Step 1: Detect Your Emotion")
if st.button("Start Webcam"):
    webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetectionTransformer)

# Music recommendation after emotion detection
st.header("Step 2: Get Music Recommendations")
if 'detected_emotion' in st.session_state:
    emotion = st.session_state['detected_emotion']
    st.write(f"Emotion Detected: {emotion}")
    
    music, similarity = load_music_data(emotion)
    song_list = music['song'].values
    selected_song = st.selectbox("Type or select a song from the dropdown", song_list)

    if st.button('Show Recommendation'):
        recommended_music_names, recommended_music_posters, recommended_music_previews, recommended_music_ids = recommend(selected_song, music, similarity)

        cols = st.columns(2)  # Create two columns for display
        for i in range(len(recommended_music_names)):
            col = cols[i % 2]
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
