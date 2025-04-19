# main.py

import streamlit as st
import tensorflow as tf
import os
from PIL import Image
from keras.models import load_model
from keras.saving import register_keras_serializable
from preprocessing.preprocess import detect_face_mtcnn, detect_face_mtcnn_suku
from model.predict_siamese import predict_suku_siamese
from tensorflow.keras.models import load_model

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# Load model tanpa error
siamese_model = load_model("model/siamese_model.h5", compile=False)

# Folder galeri yang digunakan untuk pembandingan
gallery_folder = "data/prediction/"
os.makedirs(gallery_folder, exist_ok=True)

def get_next_filename(folder, suku):
    existing_files = [f for f in os.listdir(folder) if f.startswith(suku)]
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if '_' in f and f.split('_')[1].split('.')[0].isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"{suku}_{next_number}.jpg"

def main():
    st.title("Deteksi Wajah dan Klasifikasi Suku dengan Siamese Network")
    mode = st.radio("Pilih Mode:", ["Tambah Data", "Deteksi Suku"])

    if mode == "Tambah Data":
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



    elif mode == "Deteksi Suku":
        uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_path = "temp.jpg"
            image.save(image_path)

            output_folder = f"data/temp/"
            os.makedirs(output_folder, exist_ok=True)

            _, faces, _ = detect_face_mtcnn_suku(image_path, output_folder, "temp.jpg")
            if faces:
                face = Image.open("data/temp/temp_1_cropped_1.jpg")
                label, distance = predict_suku_siamese(face, gallery_folder, siamese_model)
                if label is not None:
                    st.success(f"Suku terdeteksi: **{label.upper()}** (jarak: {distance:.4f})")
                else:
                    st.warning(
                        "Model tidak dapat mengenali suku dari wajah ini. Pastikan galeri pembanding tersedia dan jelas.")
            else:
                st.warning("Tidak ada wajah terdeteksi.")

if __name__ == "__main__":
    main()
