import cv2
import os
from mtcnn.mtcnn import MTCNN


def detect_face_mtcnn(image_path, output_folder, filename):
    # Membaca gambar
    image = cv2.imread(image_path)

    # Mendeteksi wajah menggunakan MTCNN
    detector = MTCNN()
    faces = detector.detect_faces(image)

    suku = filename.split("_")[0]
    nomor = filename.split("_")[1].split(".")[0]

    # Menyimpan gambar dengan wajah terdeteksi (opsional untuk visualisasi, bisa dihapus jika tidak dibutuhkan)
    output_image_path = os.path.join(output_folder, f"processed_{suku}_{nomor}.jpg")
    # cv2.imwrite(output_image_path, image)

    cropped_path = None

    for idx, face in enumerate(faces):
        # Mendapatkan koordinat wajah yang terdeteksi
        x, y, w, h = face['box']

        # Memotong wajah
        cropped_face = image[y:y + h, x:x + w]

        # Mengubah ukuran wajah agar sesuai dengan input model FaceNet (160x160)
        cropped_face_resized = cv2.resize(cropped_face, (160, 160))

        # Menyimpan gambar yang sudah dipotong dan diubah ukurannya
        cropped_path = os.path.join(output_folder, f"{suku}_{nomor}_cropped_{idx + 1}.jpg")
        cv2.imwrite(cropped_path, cropped_face_resized)

    return image, faces, cropped_path  # Mengembalikan gambar asli dan path dari gambar yang sudah dipotong

def detect_face_mtcnn_suku(image_path, output_folder, filename):
    import cv2
    from mtcnn import MTCNN
    import os

    image = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    # Ekstrak suku dan nomor dengan fallback
    if "_" in filename:
        parts = filename.split("_")
        suku = parts[0]
        try:
            nomor = parts[1].split(".")[0]
        except IndexError:
            nomor = "1"
    else:
        suku = filename.split(".")[0]
        nomor = "1"

    os.path.join(output_folder, f"processed_{suku}_{nomor}.jpg")
    # cv2.imwrite(output_image_path, image)

    cropped_path = None
    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        cropped_face = image[y:y + h, x:x + w]
        cropped_face_resized = cv2.resize(cropped_face, (160, 160))

        cropped_path = os.path.join(output_folder, f"{suku}_{nomor}_cropped_{idx + 1}.jpg")
        cv2.imwrite(cropped_path, cropped_face_resized)

    return image, faces, cropped_path



