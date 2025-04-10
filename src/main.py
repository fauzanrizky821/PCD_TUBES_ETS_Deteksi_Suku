import streamlit as st
from PIL import Image
import os

from preprocessing.preprocess import detect_face_mtcnn


def get_next_filename(folder, suku):
    existing_files = [f for f in os.listdir(folder) if f.startswith(suku)]
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if '_' in f]
    next_number = max(numbers, default=0) + 1
    return f"{suku}_{next_number}.jpg"


def main():
    st.title("Deteksi Wajah dan Klasifikasi Suku")

    st.header("Upload Gambar dan Pilih Suku")

    suku = st.selectbox("Pilih Suku:", ["jawa", "sunda", "batak", "ambon", "padang", "cina"])

    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        upload_folder = f"data/raw/{suku}"
        output_folder = f"data/processed/{suku}"
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        filename = get_next_filename(upload_folder, suku)
        image_path = os.path.join(upload_folder, filename)

        image.save(image_path)

        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        img, faces, output_image_path = detect_face_mtcnn(image_path, output_folder, filename)

        if len(faces) > 0:
            st.subheader(f"Wajah Terdeteksi: {len(faces)}")
            cols = st.columns(2)

            with cols[0]:
                st.image(img, caption="Hasil Deteksi Wajah", use_column_width=True)
        else:
            st.subheader("Tidak ada wajah terdeteksi.")


if __name__ == "__main__":
    main()
