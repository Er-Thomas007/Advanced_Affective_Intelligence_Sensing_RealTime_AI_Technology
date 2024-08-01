import cv2
import os
import numpy as np
import streamlit as st
from keras.preprocessing import image
from tensorflow.keras.models import load_model

# Adjust the path according to your file's location
model_path = r'D:\AI_Project\Next-Generation Emotion Recognition Integrating Deep Learning and Real-Time Analysis\ML_Model\Thomas.h5'

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except OSError as e:
    print(f"Error loading model: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title('Facial Emotion Analysis')

# Open video capture
cap = cv2.VideoCapture(0)

def detect_emotion(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        
        # Extract face ROI, resize to match model input size
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        
        # Preprocess image for model input
        img_pixels = roi_gray_resized.reshape((1, 48, 48, 1)) / 255.0

        # Predict emotion
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        predicted_emotion = emotions[max_index]

        # Display predicted emotion with different colors
        color = (0, 255, 0) if predicted_emotion == 'Happy' else (0, 0, 255)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

if st.button("Start Capture"):
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            st.error("Error: Failed to capture frame")
            break

        # Detect emotion and display the resulting frame
        frame = detect_emotion(frame)
        st.image(frame, channels="BGR", caption='Emotion Detection')

        # Check if the Stop Capture button is clicked
        if st.button("Stop Capture"):
            break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
