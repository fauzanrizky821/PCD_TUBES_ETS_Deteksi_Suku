import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================== PREPROCESS IMAGE =====================
def preprocess_face(face_image, target_size=(224, 224)):
    try:
        face_image = face_image.resize(target_size)
        face_array = np.array(face_image).astype("float32")
        face_array = preprocess_input(face_array)
        return np.expand_dims(face_array, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

# ===================== PREDICT SUKU =====================
def predict_suku_mobilenetv2(input_image, model, class_indices):
    processed_image = preprocess_face(input_image)
    if processed_image is None:
        logger.error("Failed to preprocess input image")
        return "Unknown", 0.0

    prediction = model.predict(processed_image, verbose=0)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    suku_labels = {v: k for k, v in class_indices.items()}
    predicted_suku = suku_labels.get(predicted_idx, "Unknown")

    logger.info(f"Predicted suku: {predicted_suku} (Confidence: {confidence:.4f})")
    return predicted_suku, float(confidence)
