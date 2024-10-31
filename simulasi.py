import cv2
import mediapipe as mp
import numpy as np
import random
import pickle
import time
import streamlit as st

# Deteksi lingkungan
is_local = not st.runtime.exists()

# Load model
with open("asl_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Fungsi untuk memproses gambar
def image_processed(hand_img):
    # ... (kode yang sama seperti sebelumnya)

# Streamlit UI
st.title("ASL Practice & Detection")
mode = st.radio("Choose Mode:", ["Mini-Game", "ASL Simulation"])

if is_local:
    cap = cv2.VideoCapture(0)
else:
    # Gunakan gambar statis atau video yang sudah diupload
    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file is not None:
        if uploaded_file.type.startswith('video'):
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
    else:
        st.warning("Please upload an image or video file to continue.")
        st.stop()

frame_placeholder = st.empty()
target_letter = st.empty()
score_text = st.empty()

if mode == "Mini-Game":
    st.subheader("Mini-Game: Match the ASL letter")
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    current_letter = random.choice(letters)
    score = 0
    target_letter.write(f"Target Letter: **{current_letter}**")
    
    while True:
        if is_local or (not is_local and uploaded_file.type.startswith('video')):
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Jika menggunakan gambar statis, gunakan frame yang sama berulang kali
            ret = True
        
        if ret:
            data = image_processed(frame)
            data = np.array(data).reshape(-1, 63)
            y_pred = svm.predict(data)
            
            if y_pred[0] == current_letter:
                score += 1
                score_text.write(f"**Score**: {score}")
                current_letter = random.choice(letters)
                target_letter.write(f"Target Letter: **{current_letter}**")
            
            # Tampilkan hasil
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            frame_placeholder.image(rgb_frame)
        
        if not is_local and not uploaded_file.type.startswith('video'):
            # Jika menggunakan gambar statis, hentikan loop setelah satu iterasi
            break

elif mode == "ASL Simulation":
    st.subheader("ASL Detection Simulation")
    
    while True:
        if is_local or (not is_local and uploaded_file.type.startswith('video')):
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Jika menggunakan gambar statis, gunakan frame yang sama berulang kali
            ret = True
        
        if ret:
            data = image_processed(frame)
            data = np.array(data).reshape(-1, 63)
            y_pred = svm.predict(data)
            
            cv2.putText(frame, str(y_pred[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            frame_placeholder.image(rgb_frame)
        
        if not is_local and not uploaded_file.type.startswith('video'):
            # Jika menggunakan gambar statis, hentikan loop setelah satu iterasi
            break

if is_local:
    cap.release()
