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
from streamlit_option_menu import option_menu

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# Load model tanpa error
siamese_model = load_model("model/siamese_model.keras", compile=True)
print(siamese_model)

# Folder galeri yang digunakan untuk pembandingan
gallery_folder = "data/gallery/"
os.makedirs(gallery_folder, exist_ok=True)

def get_next_filename(folder, suku):
    existing_files = [f for f in os.listdir(folder) if f.startswith(suku)]
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if '_' in f and f.split('_')[1].split('.')[0].isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"{suku}_{next_number}.jpg"


st.set_page_config(page_title="C6 - Tubes ETS PCD", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",  
        options=["Home", "Tambah Data", "Deteksi Suku dari File", "Deteksi Suku Sekarang", "Daftar Wajah Dataset",], 
        icons=["house-door", "cloud-arrow-up", "image", "camera", "folder2-open",],
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
            folder_path = f"data/raw/{suku}"
            
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
        st.subheader("🔍 Deteksi Suku dari Gambar Wajah")
        uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            image_path = "data/temp/temp.jpg"
            image.save(image_path)

            output_folder = f"data/temp/"
            os.makedirs(output_folder, exist_ok=True)

            _, faces, _ = detect_face_mtcnn_suku(image_path, output_folder, "temp.jpg")
            if faces:
                face = Image.open("data/temp/temp_1_cropped_1.jpg")
                label, distance, confidence, avg_distances, sorted_distances = predict_suku_siamese(face, gallery_folder, siamese_model)

                if label is not None:
                    st.success(f"\n\n**Suku terdeteksi: {label.upper()}** (jarak: {distance:.4f}) (confidence: {confidence:.2f})")
                else:
                    st.warning("Model tidak dapat mengenali suku dari wajah ini.")
            else:
                st.warning("Tidak ada wajah terdeteksi.")

    # --- DETEKSI SUKU DARI KAMERA ---
    elif selected == "Deteksi Suku Sekarang":
        st.subheader("📸 Deteksi Suku dari Kamera")
        

if __name__ == "__main__":
    main()
