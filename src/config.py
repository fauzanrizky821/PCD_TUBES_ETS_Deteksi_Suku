import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
model_path = "model/facenet_keras.h5"

print("TensorFlow version:", tf.__version__)
print("load_model:", load_model)


try:
    with h5py.File(model_path, 'r') as f:
        print("✅ File berhasil dibuka.")
        print("Isi grup root file:", list(f.keys()))
except Exception as e:
    print("❌ File tidak bisa dibuka. Error:", e)