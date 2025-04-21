import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from predict_suku import predict_suku_mobilenetv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log TensorFlow version and CPU info
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Num CPUs Available: {len(tf.config.list_physical_devices('CPU'))}")

# Log working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Add a test log to confirm logger is working
logger.info("Logging test: Script started successfully")

# ===================== LOAD DATASET =====================
def load_dataset(dataset_path):
    image_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    logger.info(f"Found {len(classes)} classes: {classes}")

    class_counts = {}
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        image_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        class_counts[cls] = len(image_files)
        for filename in image_files:
            image_paths.append(os.path.join(cls_path, filename))
            labels.append(class_indices[cls])

    logger.info(f"Loaded {len(image_paths)} images")
    logger.info("Class distribution:")
    for cls, count in class_counts.items():
        logger.info(f"{cls}: {count} images")

    return image_paths, labels, class_indices

# ===================== MAIN TESTING FUNCTION =====================
if __name__ == "__main__":
    try:
        # Check dataset path
        dataset_dir = 'data/processed/'
        logger.info(f"Checking dataset path: {dataset_dir}")
        logger.info(f"Absolute dataset path: {os.path.abspath(dataset_dir)}")
        if not os.path.exists(dataset_dir):
            logger.error(f"Dataset path {dataset_dir} does not exist")
            exit(1)
        logger.info(f"Dataset path {dataset_dir} exists")

        # Load the entire dataset
        image_paths, labels, class_indices = load_dataset(dataset_dir)
        inv_class_indices = {v: k for k, v in class_indices.items()}

        # Load the best trained model
        model_path = "model/mobilenetv2_best.h5"
        if not os.path.exists(model_path):
            logger.error(f"Trained model not found at {model_path}")
            exit(1)
        logger.info(f"Loading trained model from {model_path}")
        mobilenetv2_model = tf.keras.models.load_model(model_path)

        # Generate predictions for all images
        logger.info("Starting prediction on all dataset images")
        y_true = [inv_class_indices[l] for l in labels]
        y_pred = []
        confidences = []

        for idx, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            pred_suku, confidence = predict_suku_mobilenetv2(img, mobilenetv2_model, class_indices)
            y_pred.append(pred_suku)
            confidences.append(confidence)
            logger.info(f"Predicted {idx + 1}/{len(image_paths)}: {pred_suku} (Confidence: {confidence:.4f})")

        # Calculate accuracy
        correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total_samples = len(y_true)
        accuracy = correct_predictions / total_samples
        logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")

        # Classification Report
        logger.info("Classification Report:\n" + classification_report(y_true, y_pred))

        # Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred, labels=list(class_indices.keys()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_indices.keys()))
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
        plt.title("Confusion Matrix on Full Dataset")
        plt.tight_layout()
        plt.savefig("model/confusion_matrix_full_dataset.png")
        plt.close()
        logger.info("Confusion matrix saved to model/confusion_matrix_full_dataset.png")

        # Save predictions to a text file
        with open("model/predictions_full_dataset.txt", "w") as f:
            f.write("Image Path,True Label,Predicted Label,Confidence\n")
            for img_path, true_label, pred_label, conf in zip(image_paths, y_true, y_pred, confidences):
                f.write(f"{img_path},{true_label},{pred_label},{conf:.4f}\n")
        logger.info("Predictions saved to model/predictions_full_dataset.txt")

    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        raise