import numpy as np
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================== PREPROCESS FACE =====================
def preprocess_face(face_image, target_size=(160, 160)):
    try:
        face_image = face_image.resize(target_size)
        face_array = np.array(face_image).astype("float32")
        face_array = preprocess_input(face_array)  # MobileNetV2 preprocessing
        return np.expand_dims(face_array, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None


# ===================== PLOT PREDICTION DISTANCES =====================
def plot_prediction_distances(avg_distances, save_dir="model"):
    os.makedirs(save_dir, exist_ok=True)

    suku_list = sorted(avg_distances.keys())
    distances = [avg_distances[suku] for suku in suku_list]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(suku_list, distances, color='skyblue')
    plt.xlabel('Suku')
    plt.ylabel('Average Distance')
    plt.title('Average Siamese Distance per Suku')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_distances.png'))
    plt.close()
    logger.info("Prediction distance plot saved to model/prediction_distances.png")


# ===================== PREDICT SUKU SIAMESE =====================
def predict_suku_siamese(input_face_image, gallery_folder, model, distance_threshold=0.5):
    """
    Predict the 'suku' of an input face image using a Siamese network.

    Args:
        input_face_image: PIL Image object of the input face
        gallery_folder: Path to folder containing subfolders of suku images
        model: Trained Siamese model
        distance_threshold: Maximum distance for confident prediction

    Returns:
        tuple: (predicted_label, min_distance, confidence, avg_distances, sorted_distances)
    """
    # Preprocess input image
    input_face = preprocess_face(input_face_image)
    if input_face is None:
        logger.error("Failed to preprocess input face image")
        return None

    distances = {}  # Store distances per suku
    min_distance = float("inf")
    predicted_label = None
    gallery_images = []
    gallery_labels = []

    # Collect gallery images
    logger.info(f"Scanning gallery folder: {gallery_folder}")
    for suku in os.listdir(gallery_folder):
        suku_path = os.path.join(gallery_folder, suku)
        if not os.path.isdir(suku_path):
            continue

        for filename in os.listdir(suku_path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                gallery_face_path = os.path.join(suku_path, filename)
                gallery_face = Image.open(gallery_face_path).convert('RGB')
                gallery_face_processed = preprocess_face(gallery_face)
                if gallery_face_processed is not None:
                    gallery_images.append(gallery_face_processed)
                    gallery_labels.append(suku)
                    if suku not in distances:
                        distances[suku] = []

    if not gallery_images:
        logger.error("No valid gallery images found")
        return None

    logger.info(f"Found {len(gallery_images)} gallery images across {len(distances)} suku")

    # Batch predict distances
    batch_size = 32
    all_distances = []
    for i in range(0, len(gallery_images), batch_size):
        batch_images = gallery_images[i:i + batch_size]
        batch_inputs = [input_face] * len(batch_images)
        batch_predictions = model.predict(
            [np.vstack(batch_inputs), np.vstack(batch_images)],
            batch_size=len(batch_images),
            verbose=0
        )
        all_distances.extend(batch_predictions.flatten())

    # Organize distances by suku
    for distance, suku in zip(all_distances, gallery_labels):
        distances[suku].append(distance)
        if distance < min_distance:
            min_distance = distance
            predicted_label = suku
        logger.info(f"[{suku.upper()}] Predicted distance: {distance:.4f}")

    # Apply threshold
    if min_distance > distance_threshold:
        logger.warning(f"Minimum distance {min_distance:.4f} exceeds threshold {distance_threshold}. No confident prediction.")
        predicted_label = "Unknown"
        confidence = 0.0
    else:
        # Calculate average distances and confidence
        avg_distances = {suku: np.mean(vals) for suku, vals in distances.items()}
        sorted_distances = sorted(avg_distances.items(), key=lambda x: x[1])
        max_distance = max(avg_distances.values()) or 1.0
        confidence = 1.0 - (min_distance / max_distance) if max_distance > 0 else 1.0
    avg_distances = {suku: np.mean(vals) for suku, vals in distances.items()}
    sorted_distances = sorted(avg_distances.items(), key=lambda x: x[1])

    # Plot distances
    plot_prediction_distances(avg_distances)

    logger.info(
        f"Prediction complete: {predicted_label} (Confidence: {confidence:.4f}, Min Distance: {min_distance:.4f}, Sorted Distances: {sorted_distances})")
    return predicted_label, float(min_distance), float(confidence), avg_distances, sorted_distances