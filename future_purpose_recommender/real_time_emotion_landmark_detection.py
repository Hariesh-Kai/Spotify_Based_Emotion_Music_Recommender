import os
import cv2
import numpy as np
import streamlit as st
import joblib
from tensorflow import keras
import mediapipe as mp
import pandas as pd
import time

# Load the trained model and label encoder
model_path = 'D:/Music_Recommender/models/mediapipe_3emotion_model_1.h5'
label_encoder_path = 'D:/Music_Recommender/Label_Encoder/label_encoder_3_emotion.pkl'
model = keras.models.load_model(model_path)
le = joblib.load(label_encoder_path)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def draw_landmarks(image, landmarks, highlight_index=None):
    """Draw landmarks on the image and highlight a specific keypoint if specified."""
    if landmarks is not None:
        h, w, _ = image.shape
        for i in range(0, len(landmarks), 3):
            x = int(landmarks[i] * w)
            y = int(landmarks[i + 1] * h)
            color = (0, 255, 0)  # Default color for keypoints
            if highlight_index is not None and i // 3 == highlight_index:  # Highlight the clicked keypoint
                color = (255, 0, 0)  # Change color for highlighted keypoint
            cv2.circle(image, (x, y), 2, color, -1)
    return image

def extract_landmarks(image):
    """Extract landmarks from the image using MzediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        landmark_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark])
        return landmark_array.flatten()
    return None

def process_frame(frame, show_landmarks):
    """Process a single frame for emotion detection."""
    landmarks = extract_landmarks(frame)

    if landmarks is not None:
        prediction = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_class = np.argmax(prediction)
        emotion = le.inverse_transform([predicted_class])[0]

        if show_landmarks:
            frame = draw_landmarks(frame.copy(), landmarks)

        return frame, emotion, landmarks
    return frame, None, None

def real_time_emotion_detection(show_landmarks):
    """Detect emotions in real-time from webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    video_placeholder = st.empty()
    stop_button = st.button("Stop", key="stop_button")  # Add a button to stop detection
    predictions = []  # Store emotion predictions
    keypoints = []  # Store keypoints
    latest_emotion = None  # Variable to store the latest emotion
    frozen_frame = None  # Variable to store the frozen frame

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        frame, emotion, landmarks = process_frame(frame, show_landmarks)

        if landmarks is not None:
            # Store emotion and keypoints
            predictions.append(emotion)
            keypoints.append(landmarks)

            latest_emotion = emotion  # Update the latest emotion

            # Freeze the frame when an emotion is detected
            frozen_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Freeze the current frame

            # Display the frozen frame
            video_placeholder.image(frozen_frame, channels="RGB", use_column_width=True)

            # Display the latest detected emotion in the main app area
            st.write("Detected Emotion: ", latest_emotion)

            # Break the loop if an emotion is detected
            break

        # Show the current frame
        video_placeholder.image(frame, channels="RGB", use_column_width=True)

        # Check if the Stop button is pressed
        if stop_button:
            break

        time.sleep(0.1)  # Control frame rate for smoother output

    cap.release()

    # Create a DataFrame to display predictions and keypoints
    df = pd.DataFrame({
        "Predicted Emotion": predictions,
        "Keypoints": keypoints
    })

    st.write("Emotion Detection Results:")
    
    # Dropdown to select which keypoint to highlight
    if keypoints:
        selected_index = st.selectbox("Select a Keypoint to Highlight:", range(len(keypoints)))
        highlight_landmarks = keypoints[selected_index]
        # Draw the landmarks and highlight the selected keypoint
        highlighted_image = draw_landmarks(frozen_frame.copy(), highlight_landmarks, highlight_index=selected_index)
        st.image(highlighted_image, caption="Highlighted Keypoint", channels="RGB")
    
    st.dataframe(df)  # Display the DataFrame in Streamlit

# Streamlit app setup
st.title("Real-Time Emotion Detection")
st.write("This app uses a webcam to detect emotions in real-time.")

# Option to show landmarks
show_landmarks = st.checkbox("Show Landmarks on Image")

# Start the emotion detection
if st.button("Start Detection"):
    real_time_emotion_detection(show_landmarks)
