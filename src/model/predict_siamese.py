import numpy as np
from PIL import Image
import os

def preprocess_face(face_image):
    # Resize ke ukuran input model (160x160)
    face_image = face_image.resize((160, 160))
    face_array = np.array(face_image).astype("float32") / 255.0
    return np.expand_dims(face_array, axis=0)

def predict_suku_siamese(input_face_image, gallery_folder, model):
    input_face = preprocess_face(input_face_image)

    min_distance = float("inf")
    predicted_label = None

    # Loop semua wajah di galeri dan cari jarak terdekat
    for filename in os.listdir(gallery_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            suku = filename.split("_")[0]  # Ambil nama suku dari nama file
            gallery_face_path = os.path.join(gallery_folder, filename)
            gallery_face = Image.open(gallery_face_path)
            gallery_face = preprocess_face(gallery_face)

            # Prediksi jarak kemiripan wajah
            distance = model.predict([input_face, gallery_face])[0][0]

            if distance < min_distance:
                min_distance = distance
                predicted_label = suku

    return predicted_label, min_distance
