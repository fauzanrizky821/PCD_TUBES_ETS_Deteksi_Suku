import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Optimize CPU usage
tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)

# ===================== LOAD IMAGE =====================
def load_image(image_path, target_size=(160, 160)):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image = np.array(image)
        return preprocess_input(image)  # Normalize for MobileNetV2
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return np.zeros((*target_size, 3))

# ===================== BUILD BASE CNN =====================
def build_base_network(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze pre-trained layers
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
    ])
    return model

# ===================== BUILD SIAMESE =====================
def build_siamese_network(input_shape):
    input1 = layers.Input(shape=input_shape, name="input_1")
    input2 = layers.Input(shape=input_shape, name="input_2")

    base_network = build_base_network(input_shape)
    processed_input1 = base_network(input1)
    processed_input2 = base_network(input2)

    def euclidean_distance(vects):
        x, y = vects
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-10)

    distance = layers.Lambda(euclidean_distance)([processed_input1, processed_input2])
    output = layers.Dense(1, activation="sigmoid")(distance)

    model = Model(inputs=[input1, input2], outputs=output)
    logger.info("Siamese model built successfully")
    model.summary(print_fn=lambda x: logger.info(x))
    return model

# ===================== PAIR CREATION =====================
def create_pairs_from_folder(dataset_path):
    pairs = []
    labels = []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    logger.info(f"Found {len(classes)} classes in {dataset_path}")

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if 'cropped' in f and f.endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Class {class_name}: {len(image_files)} images")

        # Positive pairs
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                img1 = os.path.join(class_path, image_files[i])
                img2 = os.path.join(class_path, image_files[j])
                pairs.append([img1, img2])
                labels.append(1)

        # Negative pairs
        for other_class in classes:
            if other_class == class_name:
                continue
            other_path = os.path.join(dataset_path, other_class)
            other_images = [f for f in os.listdir(other_path) if 'cropped' in f and f.endswith(('.png', '.jpg', '.jpeg'))]
            for img1_file in image_files:
                sampled_negatives = np.random.choice(other_images, size=min(10, len(other_images)), replace=False)
                for img2_file in sampled_negatives:
                    img1 = os.path.join(class_path, img1_file)
                    img2 = os.path.join(other_path, img2_file)
                    pairs.append([img1, img2])
                    labels.append(0)

    pairs = np.array(pairs)
    labels = np.array(labels)
    logger.info(f"Created {len(pairs)} pairs: {np.sum(labels)} positive, {len(labels) - np.sum(labels)} negative")
    return pairs, labels

# ===================== DATA GENERATOR WITH AUGMENTATION =====================
class SiameseDataGenerator(Sequence):
    def __init__(self, pairs, labels, batch_size=24, target_size=(160, 160), augment=False):
        self.pairs = pairs
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(self.pairs))
        self.augment = augment
        if augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.7, 1.3],
                zoom_range=[0.8, 1.2],
                shear_range=0.1,
                fill_mode='nearest'
            )
        else:
            self.datagen = None
        logger.info(f"Data generator initialized with {len(pairs)} pairs, augment={augment}")

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X1_batch, X2_batch, y_batch = [], [], []

        for i in batch_indices:
            img1 = load_image(self.pairs[i][0], self.target_size)
            img2 = load_image(self.pairs[i][1], self.target_size)
            label = self.labels[i]

            if self.augment and self.datagen:
                seed = np.random.randint(0, 10000)
                img1 = self.datagen.random_transform(img1, seed=seed)
                img2 = self.datagen.random_transform(img2, seed=seed)

            X1_batch.append(img1)
            X2_batch.append(img2)
            y_batch.append(label)

        return {"input_1": np.array(X1_batch), "input_2": np.array(X2_batch)}, np.array(y_batch)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ===================== PLOT TRAINING HISTORY =====================
def plot_training_history(history, save_dir="model"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()
    logger.info("Accuracy plot saved to model/accuracy_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    logger.info("Loss plot saved to model/loss_plot.png")

# ===================== PREDICTION DISTRIBUTION CALLBACK =====================
class OutputDistributionCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.validation_data)
        logger.info(f"Epoch {epoch+1} - Prediction mean: {np.mean(preds):.4f}, std: {np.std(preds):.4f}")

# ===================== TRAINING =====================
if __name__ == "__main__":
    dataset_path = 'data/processed/'
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        exit(1)

    pairs, labels = create_pairs_from_folder(dataset_path)

    pos_pairs = pairs[labels == 1]
    neg_pairs = pairs[labels == 0]
    min_pairs = min(len(pos_pairs), len(neg_pairs))
    pos_pairs = pos_pairs[:min_pairs]
    neg_pairs = neg_pairs[:min_pairs]
    pairs = np.concatenate([pos_pairs, neg_pairs])
    labels = np.concatenate([np.ones(min_pairs), np.zeros(min_pairs)])
    logger.info(f"Balanced dataset: {len(pairs)} pairs (50% positive, 50% negative)")

    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels)
    logger.info(f"Train: {len(pairs_train)} pairs, Test: {len(pairs_test)} pairs")

    batch_size = 24
    epochs = 30
    input_shape = (160, 160, 3)

    train_gen = SiameseDataGenerator(pairs_train, labels_train, batch_size=batch_size, augment=True)
    test_gen = SiameseDataGenerator(pairs_test, labels_test, batch_size=batch_size, augment=False)

    siamese_model = build_siamese_network(input_shape)
    siamese_model.compile(optimizer=Adam(learning_rate=1.5e-4),
                         loss="binary_crossentropy",
                         metrics=["accuracy"])

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint("model/siamese_model_best.keras", save_best_only=True, monitor='val_accuracy'),
        OutputDistributionCallback(test_gen)
    ]

    history = siamese_model.fit(train_gen,
                                validation_data=test_gen,
                                epochs=epochs,
                                callbacks=callbacks)

    plot_training_history(history)

    os.makedirs("model", exist_ok=True)
    siamese_model.save("model/siamese_model.keras")
    logger.info("✅ Model saved to model/siamese_model.keras")

    test_loss, test_accuracy = siamese_model.evaluate(test_gen)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")