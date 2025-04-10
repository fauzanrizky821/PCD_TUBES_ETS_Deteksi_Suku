import cv2
import numpy as np
import os
from PIL import Image

def augment_image(image, image_name, output_folder="data/processed"):
    augmented_images = []

    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    augmented_images.append(cv2.flip(image_cv, 1))

    rows, cols, _ = image_cv.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotated = cv2.warpAffine(image_cv, M, (cols, rows))
    augmented_images.append(rotated)

    brightened = cv2.convertScaleAbs(image_cv, alpha=1, beta=50)
    augmented_images.append(brightened)

    os.makedirs(output_folder, exist_ok=True)

    augmented_image_paths = []
    for i, img in enumerate(augmented_images):
        augmented_image_path = f"{output_folder}/processed_{image_name}_augmented_{i + 1}.jpg"
        cv2.imwrite(augmented_image_path, img)
        augmented_image_paths.append(augmented_image_path)

    augmented_images_pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in augmented_images]

    return augmented_images_pil, augmented_image_paths


