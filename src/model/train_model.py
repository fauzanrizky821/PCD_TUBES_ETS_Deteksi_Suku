import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
from sklearn.model_selection import train_test_split
from PIL import Image


# Fungsi untuk membangun base CNN
def build_base_network(input_shape):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    return model


# Fungsi untuk membangun model Siamese
def build_siamese_network(input_shape):
    input1 = layers.Input(shape=input_shape)
    input2 = layers.Input(shape=input_shape)

    base_network = build_base_network(input_shape)

    processed_input1 = base_network(input1)
    processed_input2 = base_network(input2)

    @register_keras_serializable()
    def euclidean_distance(vects):
        x, y = vects
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

    distance = layers.Lambda(euclidean_distance, output_shape=(1,))([processed_input1, processed_input2])

    output = layers.Dense(1, activation="sigmoid")(distance)

    model = Model(inputs=[input1, input2], outputs=output)
    return model


# Fungsi untuk memuat gambar dan ubah jadi array numpy
def load_image(image_path, target_size=(160, 160)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return np.array(image) / 255.0  # Normalisasi


# Membuat pasangan data dari folder dataset
def create_pairs_from_folder(dataset_path):
    pairs = []
    labels = []

    classes = os.listdir(dataset_path)
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if 'cropped' in f]

        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                img1_path = os.path.join(class_path, image_files[i])
                img2_path = os.path.join(class_path, image_files[j])
                pairs.append([img1_path, img2_path])
                labels.append(1)

        for other_class in classes:
            if other_class == class_name:
                continue
            other_path = os.path.join(dataset_path, other_class)
            other_images = [f for f in os.listdir(other_path) if 'cropped' in f]

            for img1_file in image_files:
                for img2_file in other_images:
                    img1_path = os.path.join(class_path, img1_file)
                    img2_path = os.path.join(other_path, img2_file)
                    pairs.append([img1_path, img2_path])
                    labels.append(0)

    return np.array(pairs), np.array(labels)


# === MAIN ===
dataset_path = 'data/processed/'  # folder gambar yang sudah diproses dan cropped
pairs, labels = create_pairs_from_folder(dataset_path)

X1 = np.array([load_image(p[0]) for p in pairs])
X2 = np.array([load_image(p[1]) for p in pairs])

# Bagi data
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, labels, test_size=0.2, random_state=42)

# Buat dan latih model
siamese_model = build_siamese_network(input_shape=(160, 160, 3))
siamese_model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
siamese_model.fit([X1_train, X2_train], y_train, epochs=100, batch_size=32, validation_split=0.1)

# Simpan model
siamese_model.save("model/siamese_model.h5")
print("✅ Model saved to model/siamese_model.h5")
