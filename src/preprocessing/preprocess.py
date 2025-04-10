import os

import cv2
from mtcnn.mtcnn import MTCNN

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



def detect_face_mtcnn(image_path, output_folder, filename):
    image = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    suku = filename.split("_")[0]
    nomor = filename.split("_")[1].split(".")[0]

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_image_path = os.path.join(output_folder, f"processed_{suku}_{nomor}.jpg")
    cv2.imwrite(output_image_path, image)

    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        cropped_face = image[y:y + h, x:x + w]
        cropped_path = os.path.join(output_folder, f"{suku}_{nomor}_cropped_{idx + 1}.jpg")
        cv2.imwrite(cropped_path, cropped_face)

    return image, faces, output_image_path
