import numpy as np
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================== PREDICT SUKU WITH CLASSIFICATION =====================
def preprocess_face(face_image, target_size=(224, 224)):
    """
    Preprocess face image for model prediction.
    Args:
        face_image (PIL.Image): Input face image to be processed.
        target_size (tuple): Desired output size of image (default: (224, 224)).

    Returns:
        np.array: Preprocessed image ready for model prediction.
    """
    try:
        face_image = face_image.resize(target_size)
        face_array = np.array(face_image).astype("float32")
        face_array = preprocess_input(face_array)  # Untuk MobileNetV2
        return np.expand_dims(face_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_suku(face_image, model, class_labels):
    """
    Predict suku label based on face image using MobileNetV2 model.

    Args:
        input_face_image (PIL.Image): The input face image.
        model (tf.keras.Model): Pretrained classification model.
        class_labels (list): List of class labels for suku.
        target_size (tuple): The target image size (default: (224, 224)).

    Returns:
        str: Predicted suku label.
        float: Confidence of prediction.
    """
    # Preprocess input image
    processed_face = preprocess_face(face_image)
    if processed_face is None:
        return None, 0.0

    prediction = model.predict(processed_face)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_label = class_labels[predicted_index]

    return predicted_label, confidence


# ===================== LOAD MODEL AND CLASS LABELS =====================
def load_model_and_labels(model_path, label_file_path):
    """
    Load pretrained model and corresponding class labels for prediction.

    Args:
        model_path (str): Path to the saved model.
        label_file_path (str): Path to the file containing class labels.

    Returns:
        tf.keras.Model: Loaded model.
        list: List of class labels.
    """
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load class labels
    with open(label_file_path, 'r') as f:
        class_labels = f.readlines()
    class_labels = [label.strip() for label in class_labels]

    return model, class_labels


# ===================== TEST THE MODEL =====================
def test_predict_suku():
    # Example paths
    model_path = 'path_to_your_model.h5'
    label_file_path = 'path_to_class_labels.txt'
    input_image_path = 'path_to_input_face_image.jpg'

    # Load model and labels
    model, class_labels = load_model_and_labels(model_path, label_file_path)

    # Load input face image
    input_face_image = Image.open(input_image_path).convert('RGB')

    # Predict suku
    predicted_suku, confidence = predict_suku(input_face_image, model, class_labels)

    if predicted_suku:
        print(f"Predicted Suku: {predicted_suku} with Confidence: {confidence:.2f}")
    else:
        print("Prediction failed.")


# Run the test
test_predict_suku()
