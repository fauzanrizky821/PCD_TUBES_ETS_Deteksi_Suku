import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from mtcnn import MTCNN  # This is from mtcnn package
import torch
from facenet_pytorch import InceptionResnetV1
import logging
import tempfile
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load MTCNN detector for face detection
detector = MTCNN()

# Load FaceNet model for embedding extraction
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def detect_and_crop_face(image_path, output_dir="cropped_faces"):
    try:
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                logger.error("Gagal membaca gambar dari path")
                return None, None
        else:
            pil_image = Image.open(image_path).convert("RGB")
            image = np.array(pil_image)
        
        image_rgb = image  # Sudah RGB dari PIL

        faces = detector.detect_faces(image_rgb)
        if not faces:
            logger.error("Tidak ada wajah terdeteksi di gambar")
            return None, None
        
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        w, h = abs(w), abs(h)

        face = image_rgb[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)

        os.makedirs(output_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir=output_dir) as tmp:
            output_path = tmp.name
            face_pil.save(output_path)
            logger.info(f"Gambar wajah disimpan di: {output_path}")
        
        return face_pil, output_path
    
    except Exception as e:
        logger.error(f"Error saat deteksi wajah: {e}")
        return None, None


def preprocess_face(face_image, target_size=(160, 160)):
    """
    Preprocessing gambar wajah untuk FaceNet.
    """
    try:
        if face_image is None:
            return None
            
        # Resize to the required input size for FaceNet
        face_image = face_image.resize(target_size)
        face_tensor = torch.from_numpy(np.array(face_image)).float()
        
        # Convert to torch tensor with correct format for FaceNet
        face_tensor = face_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
        face_tensor = face_tensor / 255.0  # Normalize to [0, 1]
        
        # Apply normalization used in FaceNet
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for i in range(3):
            face_tensor[i] = (face_tensor[i] - mean[i]) / std[i]
            
        logger.info("Gambar wajah berhasil diproses untuk model.")
        return face_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def get_embedding(face_tensor):
    """
    Get face embedding using FaceNet model
    """
    try:
        if face_tensor is None:
            return None
        
        # Get embedding (feature vector)
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(device))
        
        return embedding.cpu().numpy()[0]
    
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def calculate_similarity(emb1, emb2):
    """
    Calculate cosine similarity and Euclidean distance between two embeddings
    """
    if emb1 is None or emb2 is None:
        return 0, float('inf')

    # Normalize embeddings for cosine similarity
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)

    cosine_similarity = np.dot(emb1_norm, emb2_norm)

    # Euclidean distance (tanpa normalisasi)
    euclidean_distance = np.linalg.norm(emb1 - emb2)

    return cosine_similarity, euclidean_distance


def evaluate_roc_curve(similarity_scores, labels):
    """
    Evaluasi threshold optimal menggunakan ROC curve
    similarity_scores: list of similarity scores or distances
    labels: true labels (1 if match, 0 if not match)
    """
    fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
    roc_auc = auc(fpr, tpr)

    # Gambar ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Menampilkan di Streamlit
    st.pyplot(plt)

    # Mengembalikan threshold optimal
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    return optimal_threshold


def compare_faces(img1, img2, threshold=0.7):
    """
    Compare two faces and determine if they belong to the same person
    """
    output_dir = "cropped_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect and crop faces
    face_image1, cropped_path1 = detect_and_crop_face(img1, output_dir)
    face_image2, cropped_path2 = detect_and_crop_face(img2, output_dir)

    
    if face_image1 is None or face_image2 is None:
        return {
            'similarity_score': 0.0,
            'is_match': False,
            'distance': 0.0,
            'decision': "Face detection failed for one or both images",
            'cropped_path1': cropped_path1,
            'cropped_path2': cropped_path2
        }
    
    # Preprocess faces
    face1_tensor = preprocess_face(face_image1)
    face2_tensor = preprocess_face(face_image2)
    
    if face1_tensor is None or face2_tensor is None:
        return {
            'similarity_score': 0.0,
            'is_match': False,
            'distance': 0.0,
            'decision': "Failed to preprocess faces",
            'cropped_path1': cropped_path1,
            'cropped_path2': cropped_path2
        }
    
    # Get embeddings
    emb1 = get_embedding(face1_tensor)
    emb2 = get_embedding(face2_tensor)
    
    if emb1 is None or emb2 is None:
        return {
            'similarity_score': 0.0,
            'is_match': False,
            'distance': 0.0,
            'decision': "Failed to extract embeddings",
            'cropped_path1': cropped_path1,
            'cropped_path2': cropped_path2
        }
    
    # Calculate similarity
    similarity_score, distance = calculate_similarity(emb1, emb2)
    is_match = similarity_score >= threshold

    
    return {
        'similarity_score': float(similarity_score),
        'is_match': is_match,
        'distance': float(distance),
        'decision': "Wajah adalah orang yang SAMA." if is_match else "Wajah adalah orang yang BERBEDA.",
        'cropped_path1': cropped_path1,
        'cropped_path2': cropped_path2
    }
