# main.py
import cv2
import streamlit as st
import tensorflow as tf
import os
import time
import uuid
from PIL import Image
import logging
from keras.models import load_model
from keras.saving import register_keras_serializable
from preprocessing.preprocess import detect_face_mtcnn, detect_face_mtcnn_suku
from model.predict_siamese import predict_suku_siamese
from model.face_similarity import compare_faces
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from model.predict_suku import predict_suku_mobilenetv2


def get_next_filename(folder, suku):
    existing_files = [f for f in os.listdir(folder) if f.startswith(suku)]
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if '_' in f and f.split('_')[1].split('.')[0].isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"{suku}_{next_number}.jpg"


st.set_page_config(page_title="C6 - Tubes ETS PCD", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",  
        options=["Home", "Tambah Data", "Deteksi Suku dari File", "Deteksi Suku Sekarang", "Daftar Wajah Dataset", "Face Similarity"], 
        icons=["house-door", "cloud-arrow-up", "image", "camera", "folder2-open", "person-check-fill"],
        menu_icon="menu-app",  
        default_index=0,
        orientation="vertical" 
    )

def main():

    st.title("Pengembangan Sistem Pengenalan Wajah dan Deteksi Suku Menggunakan Computer Vision")

    # --- HOME ---
    if selected == "Home":
        st.subheader("👥 Kelompok C6 - Pengolahan Citra Digital")
        cols = st.columns(3)
        anggota = [
            {"Nama": "Fauzan Rizky R.", "NIM": "231511076", "foto": "src/images/wildan.jpg"},
            {"Nama": "Muhammad Wildan G.", "NIM": "231511087", "foto": "src/images/wildan.jpg"},
            {"Nama": "Restu Akbar", "NIM": "231511088", "foto": "src/images/wildan.jpg"},
        ]

        cols = st.columns(3)

        for i, col in enumerate(cols):
            with col:
                st.image(anggota[i]["foto"], width=150)
                st.markdown(f"**{anggota[i]['Nama']}**")
                st.markdown(f"**{anggota[i]['NIM']}**")


        # st.image("images/foto_kelompok.png", caption="Kelompok C6", use_column_width=True)


        st.markdown("""
        <div style='background-color: #383838; padding: 20px; border-radius: 10px;'>
            <h3>Deskripsi Aplikasi</h3>
            <p>
                Aplikasi ini dibuat untuk mendeteksi wajah dan mengklasifikasikan etnis/suku pengguna berdasarkan citra wajah.
                Model yang digunakan adalah Siamese Network dengan MTCNN untuk deteksi wajah.
            </p>
            <p>
                Fitur utama aplikasi ini yaitu: Menambah data wajah berdasarkan suku, Mendeteksi wajah dari gambar dan mengklasifikasikannya ke suku paling mirip, dan Mendeteksi Wajah dari kamera dan mengklasifikasikannya ke suku paling mirip.
            </p>
        </div>
        """, unsafe_allow_html=True)


    # --- TAMBAH DATA ---
    elif selected == "Tambah Data":
        st.subheader("📥 Tambah Data Wajah ke Dataset")
        suku = st.selectbox("Pilih Suku:", ["jawa", "sunda", "batak", "ambon", "padang", "cina"])
        uploaded_files = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            success_count = 0
            failure_count = 0

            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)

                upload_folder = f"data/raw/{suku}"
                output_folder = f"data/processed/{suku}"
                os.makedirs(upload_folder, exist_ok=True)
                os.makedirs(output_folder, exist_ok=True)

                filename = get_next_filename(upload_folder, suku)
                image_path = os.path.join(upload_folder, filename)
                image.save(image_path)

                _, faces, _ = detect_face_mtcnn(image_path, output_folder, filename)

                if len(faces) > 0:
                    success_count += 1
                else:
                    failure_count += 1

            if success_count > 0:
                st.success(f"{success_count} gambar berhasil diproses dan wajah terdeteksi.")
            if failure_count > 0:
                st.warning(f"{failure_count} gambar tidak berhasil diproses atau tidak ada wajah yang terdeteksi.")

    # --- DAFTAR DATA ---
    elif selected == "Daftar Wajah Dataset":
        st.subheader("📂 Daftar Data Wajah per Suku")
        daftar_suku = ["jawa", "sunda", "batak", "ambon", "padang", "cina"]
        
        for suku in daftar_suku:
            st.markdown(f"### {suku.capitalize()}")
            folder_path = f"data/processed/{suku}"
            
            if os.path.exists(folder_path):
                images = os.listdir(folder_path)
                cols = st.columns(5) 
                for i, img_name in enumerate(images):
                    img_path = os.path.join(folder_path, img_name)
                    try:
                        with cols[i % 5]:
                            st.image(img_path, caption=img_name, width=100)
                    except:
                        pass
            else:
                st.warning(f"Tidak ditemukan folder untuk suku: {suku}")

    # --- DETEKSI SUKU DARI FILE ---
    elif selected == "Deteksi Suku dari File":
        def generate_class_indices(dataset_path):
            suku_folders = sorted(os.listdir(dataset_path))  # sort biar konsisten
            class_indices = {suku: idx for idx, suku in enumerate(suku_folders)}
            return class_indices

        class_indices = generate_class_indices("data/processed/")
        print(class_indices)

        # Load model MobileNetV2
        mobilenet_model = load_model("model/mobilenetv2_final_finetuned.h5", compile=True)
        print(mobilenet_model)

        st.subheader("🔍 Deteksi Suku dari Gambar Wajah")
        uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", width=300)

            image_path = "data/temp/temp.jpg"
            image.save(image_path)

            output_folder = f"data/temp/"
            os.makedirs(output_folder, exist_ok=True)

            _, faces, _ = detect_face_mtcnn_suku(image_path, output_folder, "temp.jpg")
            if faces:
                face = Image.open("data/temp/temp_1_cropped_1.jpg")
                predicted_suku, confidence = predict_suku_mobilenetv2(face, mobilenet_model, class_indices)

                if predicted_suku is not None:
                    st.success(f"\n\n**Suku terdeteksi: {predicted_suku.upper()}** (confidence: {confidence:.2f})")
                else:
                    st.warning("Model tidak dapat mengenali suku dari wajah ini.")
            else:
                st.warning("Tidak ada wajah terdeteksi.")

    # --- DETEKSI SUKU DARI KAMERA ---
    elif selected == "Deteksi Suku Sekarang":
        def generate_class_indices(dataset_path):
            suku_folders = sorted(os.listdir(dataset_path))  # sort biar konsisten
            class_indices = {suku: idx for idx, suku in enumerate(suku_folders)}
            return class_indices

        class_indices = generate_class_indices("data/processed/")

        # Load model MobileNetV2
        mobilenet_model = load_model("model/mobilenetv2_final_finetuned.h5", compile=True)

        st.subheader("📸 Deteksi Suku dari Kamera")

        capture_btn = st.button("Aktifkan Kamera dan Deteksi")

        if capture_btn:
            st.warning("Tunggu sebentar, sedang mengakses kamera...")

            # --- AKSES KAMERA DAN TANGKAP FRAME ---
            cap = cv2.VideoCapture(0)  # 0 = kamera default
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Gagal mengakses kamera.")
            else:
                # Simpan frame sebagai file temporer
                temp_image_path = "data/temp/webcam_temp.jpg"
                os.makedirs("data/temp", exist_ok=True)
                cv2.imwrite(temp_image_path, frame)

                st.image(frame, channels="BGR", caption="Gambar dari Kamera", width=300)

                # Deteksi wajah dan prediksi
                _, faces, _ = detect_face_mtcnn_suku(temp_image_path, "data/temp", "webcam_temp.jpg")

                if faces:
                    face = Image.open("data/temp/webcam_temp_1_cropped_1.jpg")
                    predicted_suku, confidence = predict_suku_mobilenetv2(face, mobilenet_model, class_indices)

                    if predicted_suku is not None:
                        st.success(f"**Suku terdeteksi: {predicted_suku.upper()}** (confidence: {confidence:.2f})")
                    else:
                        st.warning("Model tidak dapat mengenali suku dari wajah ini.")
                else:
                    st.warning("Tidak ada wajah terdeteksi dari gambar kamera.")
        

    elif selected == "Face Similarity":
        st.title("Face Similarity Detection with FaceNet")
        
        # Create directory for cropped faces
        os.makedirs("cropped_faces", exist_ok=True)

        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Upload images
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file1 = st.file_uploader("Choose first image", type=["jpg", "jpeg", "png"])
            if uploaded_file1 is not None:
                image1 = Image.open(uploaded_file1)
                st.image(image1, caption="Image 1", use_column_width=True)
        
        with col2:
            uploaded_file2 = st.file_uploader("Choose second image", type=["jpg", "jpeg", "png"])
            if uploaded_file2 is not None:
                image2 = Image.open(uploaded_file2)
                st.image(image2, caption="Image 2", use_column_width=True)
        
        # Similarity threshold
        threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        if st.button("Compare Faces"):
            if uploaded_file1 is None or uploaded_file2 is None:
                st.error("Please upload both images")
            else:
                with st.spinner("Processing..."):
                    # Compare faces
                    result = compare_faces(uploaded_file1, uploaded_file2, threshold)
                    
                    # Display results
                    st.subheader("Results")
                    st.write(f"Similarity Score: {result['similarity_score']:.4f}")
                    st.write(f"Distance: {result['distance']:.4f}")
                    st.write(f"Decision: {result['decision']}")
                    
                    # Display cropped faces if available
                    if result['cropped_path1'] and result['cropped_path2']:
                        st.subheader("Detected Faces")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result['cropped_path1'], caption="Face 1", use_column_width=True)
                        with col2:
                            st.image(result['cropped_path2'], caption="Face 2", use_column_width=True)
                    
                    if result['is_match']:
                        st.success("Jadi, kedua wajah tersebut adalah orang yang sama")
                    else:
                        st.warning("jadi, kedua wajah tersebut bukan orang yang sama")

                    # Simpan ke histori
            st.session_state.history.append(result)

        # Tampilkan histori
        if st.session_state.history:
            st.subheader("History of Comparisons")
            for i, item in enumerate(reversed(st.session_state.history)):
                st.markdown(f"**Comparison #{len(st.session_state.history)-i}**")
                cols = st.columns(2)
                with cols[0]:
                    st.image(item["cropped_path1"], caption="Face 1", use_column_width=True)
                with cols[1]:
                    st.image(item["cropped_path2"], caption="Face 2", use_column_width=True)
                st.write(f"Similarity Score: {item['similarity_score']:.4f}")
                st.write(f"Distance: {item['distance']:.4f}")
                st.write(f"Decision: {item['decision']}")
                if item['is_match']:
                    st.success("Match!")
                else:
                    st.warning("Not a match!")
                st.markdown("---")

        # ROC Curve dari histori
        if len(st.session_state.history) >= 2:
            st.subheader("ROC Curve from History")
            scores = [h["similarity_score"] for h in st.session_state.history]
            labels = [1 if h["is_match"] else 0 for h in st.session_state.history]

            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
