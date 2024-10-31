import cv2
import mediapipe as mp
import numpy as np
import random
import pickle
import time
import streamlit as st
import tempfile

# Deteksi lingkungan
is_local = not st.runtime.exists()

# Load the SVM model for ASL letter detection
with open("asl_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Set up MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7
)

# Streamlit sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["ASL Game & Simulation", "About ASL", "Creators"])

# Fungsi untuk memproses gambar
def image_processed(hand_img):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)
    output = hands.process(img_flip)
    try:
        data = output.multi_hand_landmarks[0]
        data = str(data).strip().split("\n")
        garbage = ["landmark {", "  visibility: 0.0", "  presence: 0.0", "}"]
        clean = [float(i.strip()[2:]) for i in data if i not in garbage]
        return clean
    except:
        return np.zeros([1, 63], dtype=int)[0]

# Halaman Utama: Mini-Game ASL dan Simulasi
if page == "ASL Game & Simulation":
    st.title("ASL Practice & Detection")
    st.write("Select an option to start:")

    # Tombol untuk memilih mode
    mode = st.radio("Choose Mode:", ["Mini-Game", "ASL Simulation"])

    # Webcam placeholder
    frame_placeholder = st.empty()
    target_letter = st.empty()
    score_text = st.empty()

    # Inisialisasi input
    if is_local:
        cap = cv2.VideoCapture(0)
    else:
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

    # Mode Mini-Game
    if mode == "Mini-Game":
        st.subheader("Mini-Game: Match the ASL letter")
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        current_letter = random.choice(letters)
        score = 0

        # Display target letter and score
        target_letter.write(f"Target Letter: **{current_letter}**")

        correct_display_time = 0  # Waktu untuk menampilkan pesan "Correct!"

        # Loop untuk Mini-Game
        while True:
            if is_local or (not is_local and uploaded_file.type.startswith('video')):
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                ret = True
            
            if ret:
                # Prediksi huruf dari input
                data = image_processed(frame)
                data = np.array(data).reshape(-1, 63)
                y_pred = svm.predict(data)

                # Periksa apakah huruf yang terdeteksi cocok
                if y_pred[0] == current_letter:
                    score += 1
                    score_text.write(f"**Score**: {score}")
                    current_letter = random.choice(letters)
                    target_letter.write(f"Target Letter: **{current_letter}**")
                    correct_display_time = time.time()  # Simpan waktu ketika tebakan benar

                # Tampilkan pesan "Correct!" selama 1 detik setelah tebakan benar
                if time.time() - correct_display_time < 1:
                    cv2.putText(
                        frame,
                        "Correct!",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )

                # Tampilkan hasil
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=2
                            ),
                            mp_drawing.DrawingSpec(
                                color=(255, 255, 255), thickness=2, circle_radius=0
                            ),
                        )

                # Tampilkan frame di Streamlit
                frame_placeholder.image(rgb_frame)

            if not is_local and not uploaded_file.type.startswith('video'):
                # Jika menggunakan gambar statis, hentikan loop setelah satu iterasi
                break

    # Mode Simulasi Deteksi
    elif mode == "ASL Simulation":
        st.subheader("ASL Detection Simulation")

        # Loop untuk simulasi deteksi ASL
        while True:
            if is_local or (not is_local and uploaded_file.type.startswith('video')):
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                ret = True
            
            if ret:
                # Proses dan prediksi huruf
                data = image_processed(frame)
                data = np.array(data).reshape(-1, 63)
                y_pred = svm.predict(data)

                # Tampilkan huruf prediksi pada frame
                cv2.rectangle(frame, (0, 0), (230, 70), (245, 117, 16), -1)
                cv2.putText(
                    frame,
                    str(y_pred[0]),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                )

                # Gambar landmark tangan
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=2
                            ),
                            mp_drawing.DrawingSpec(
                                color=(255, 255, 255), thickness=2, circle_radius=0
                            ),
                        )

                # Tampilkan hasil pada Streamlit
                frame_placeholder.image(rgb_frame)

            if not is_local and not uploaded_file.type.startswith('video'):
                # Jika menggunakan gambar statis, hentikan loop setelah satu iterasi
                break

    if is_local:
        cap.release()
        cv2.destroyAllWindows()

# Halaman Kedua: Tentang ASL
elif page == "About ASL":
    st.title("About American Sign Language (ASL)")
    st.write(
        "American Sign Language (ASL) is a visual language that serves as the primary language for the Deaf community in the United States and parts of Canada."
    )
    st.image("asl-alphabet.jpg", caption="Example of ASL Alphabet")

# Halaman Ketiga: Pembuat
elif page == "Creators":
    st.title("Creators")
    st.write("This application was created by a team of dedicated developers.")
    col1, col2 = st.columns(2)

    with col1:
        st.image("p1.png", width=150)
        st.write("**Name**: Creator 1")
        st.write("**NIM**: 123456789")
        st.write("**Class**: Class A")

        st.image("p2.png", width=150)
        st.write("**Name**: Creator 2")
        st.write("**NIM**: 987654321")
        st.write("**Class**: Class B")

    with col2:
        st.image("p3.png", width=150)
        st.write("**Name**: Creator 3")
        st.write("**NIM**: 112233445")
        st.write("**Class**: Class C")

        st.image("p4.png", width=150)
        st.write("**Name**: Creator 4")
        st.write("**NIM**: 556677889")
        st.write("**Class**: Class D")
