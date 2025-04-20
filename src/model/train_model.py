import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

# Disable GPU (already set for CPU-only)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disable oneDNN optimizations to avoid potential CPU issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Optimize CPU threads for 12-thread CPU
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log TensorFlow version and CPU info
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Num CPUs Available: {len(tf.config.list_physical_devices('CPU'))}")

# Optional memory logging (requires psutil)
try:
    import psutil
    def log_memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
except ImportError:
    def log_memory_usage():
        logger.info("Memory usage logging skipped (psutil not installed)")

# Log working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Add a test log to confirm logger is working
logger.info("Logging test: Script started successfully")

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

# ===================== AUGMENT AND SAVE IMAGES =====================
def augment_and_save_images(image_paths, labels, class_indices, augment_dir, num_augmented=5):
    """
    Augment training images and save them to a new directory.
    Returns the updated list of image paths and labels.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    inv_class_indices = {v: k for k, v in class_indices.items()}
    augmented_image_paths = image_paths.copy()
    augmented_labels = labels.copy()

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        class_name = inv_class_indices[label]
        class_dir = os.path.join(augment_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

        # Generate augmented images
        for i in range(num_augmented):
            augmented_iter = datagen.flow(img_array, batch_size=1)
            augmented_img = next(augmented_iter)[0].astype(np.uint8)
            augmented_img = Image.fromarray(augmented_img)

            # Save augmented image
            base_filename = os.path.basename(img_path)
            name, ext = os.path.splitext(base_filename)
            aug_filename = f"{name}_aug_{i}{ext}"
            aug_path = os.path.join(class_dir, aug_filename)
            augmented_img.save(aug_path)

            augmented_image_paths.append(aug_path)
            augmented_labels.append(label)

        logger.info(f"Augmented {idx + 1}/{len(image_paths)} images for class {class_name}")

    logger.info(f"Total images after augmentation: {len(augmented_image_paths)}")
    return augmented_image_paths, augmented_labels

# ===================== BUILD MOBILENETV2 CLASSIFIER =====================
def build_mobilenetv2_classifier(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='suku_output')(x)

    model = Model(inputs, outputs)
    logger.info("MobileNetV2 classifier built successfully")
    model.summary(print_fn=lambda x: logger.info(x))
    return model

# ===================== DATA GENERATOR =====================
class CNNDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, class_indices, batch_size=32, target_size=(224, 224), augment=False, **kwargs):
        super().__init__(**kwargs)
        self.image_paths = image_paths
        self.labels = labels
        self.class_indices = class_indices
        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(self.image_paths), dtype=np.int64)
        self.augment = augment
        if augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                brightness_range=[0.7, 1.3],
                fill_mode='nearest'
            )
        else:
            self.datagen = None
        logger.info(f"Data generator initialized with {len(image_paths)} images, augment={augment}")

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, y_batch = [], []

        for i in batch_indices:
            img_path = self.image_paths[i]
            label = self.labels[i]
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size)
            img_array = np.array(img).astype("float32")
            img_array = preprocess_input(img_array)

            if self.augment and self.datagen:
                img_array = self.datagen.random_transform(img_array)

            X_batch.append(img_array)
            y_batch.append(label)

        return np.array(X_batch), tf.keras.utils.to_categorical(y_batch, num_classes=len(self.class_indices))

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

def get_class_weights(labels, class_indices):
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = {class_idx: weight for class_idx, weight in zip(class_indices.values(), class_weights)}
    return class_weight_dict

# ===================== TRAINING AND INFERENCE =====================
if __name__ == "__main__":
    try:
        # Hyperparameters
        batch_size = 32  # Increased for 16 GB RAM
        epochs = 10      # Increased for better convergence
        learning_rate = 1e-4
        image_size = (224, 224)
        num_augmented_per_image = 5  # Number of augmented images per original image

        # Check dataset path
        dataset_dir = 'data/processed/'
        logger.info(f"Checking dataset path: {dataset_dir}")
        logger.info(f"Absolute dataset path: {os.path.abspath(dataset_dir)}")
        if not os.path.exists(dataset_dir):
            logger.error(f"Dataset path {dataset_dir} does not exist")
            exit(1)
        logger.info(f"Dataset path {dataset_dir} exists")

        # Load dataset
        image_paths, labels, class_indices = load_dataset(dataset_dir)
        inv_class_indices = {v: k for k, v in class_indices.items()}

        # Stratified split: Train, Validation, Test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.1, stratify=train_labels, random_state=42
        )
        logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

        # Augment the training set and save to disk
        augment_dir = 'data/augmented_train/'
        logger.info(f"Augmenting training set and saving to {augment_dir}")
        train_paths, train_labels = augment_and_save_images(
            train_paths, train_labels, class_indices, augment_dir, num_augmented=num_augmented_per_image
        )
        logger.info(f"Training set after augmentation: {len(train_paths)} images")

        # Generators
        train_gen = CNNDataGenerator(train_paths, train_labels, class_indices, batch_size=batch_size, augment=True)
        val_gen = CNNDataGenerator(val_paths, val_labels, class_indices, batch_size=batch_size, augment=False)
        test_gen = CNNDataGenerator(test_paths, test_labels, class_indices, batch_size=batch_size, augment=False)

        # Log memory usage after loading data
        log_memory_usage()

        # Build model
        mobilenetv2_model = build_mobilenetv2_classifier(num_classes=len(class_indices), input_shape=(*image_size, 3))

        # Compile
        mobilenetv2_model.compile(optimizer=Adam(learning_rate),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

        # Log memory usage after building model
        log_memory_usage()

        # Callbacks
        os.makedirs("model", exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint("model/mobilenetv2_best.h5", save_best_only=True)
        ]

        # Class weights (recompute after augmentation)
        class_weight_dict = get_class_weights(train_labels, class_indices)
        logger.info(f"Class weights after augmentation: {class_weight_dict}")

        # Train model
        logger.info("Starting training")
        history = mobilenetv2_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )

        # Log memory usage after training
        log_memory_usage()

        # Save final model
        mobilenetv2_model.save("model/mobilenetv2_final.h5")
        logger.info("Final model saved to model/mobilenetv2_final.h5")

        # Plot training history
        plot_training_history(history)

        # Evaluation on test set
        loss, acc = mobilenetv2_model.evaluate(test_gen)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

        # Confusion Matrix
        y_true = [inv_class_indices[l] for l in test_labels]
        y_pred = []
        for p in test_paths:
            img = Image.open(p).convert('RGB')
            pred_suku, _ = predict_suku_mobilenetv2(img, mobilenetv2_model, class_indices)
            y_pred.append(pred_suku)

        logger.info(f"Unique y_true: {set(y_true)}")
        logger.info(f"Unique y_pred: {set(y_pred)}")
        logger.info(f"All class labels: {list(class_indices.keys())}")
        logger.info(f"Jumlah y_true: {len(y_true)}, Jumlah y_pred: {len(y_pred)}")

        # Classification Report
        logger.info("Classification Report:\n" + classification_report(y_true, y_pred))

        # Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred, labels=list(class_indices.keys()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_indices.keys()))
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
        plt.title("Confusion Matrix on Test Set")
        plt.tight_layout()
        plt.savefig("model/confusion_matrix.png")
        plt.close()
        logger.info("Confusion matrix saved to model/confusion_matrix.png")
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        raise