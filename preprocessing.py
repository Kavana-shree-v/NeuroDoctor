"""
Brain Tumor MRI — Image Preprocessing Pipeline
Uses CLAHE enhancement, resizing, and normalization.
Dataset: Training/Testing split with 4 classes:
  glioma | meningioma | notumor | pituitary
"""

import os
import numpy as np
import cv2
import tensorflow as tf

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
TRAIN_DIR = os.path.join("dataset", "Training")
TEST_DIR  = os.path.join("dataset", "Testing")

# ─── CLAHE Enhancement ────────────────────────────────────────────────────────
def enhance_image(image):
    """Apply CLAHE contrast enhancement to an MRI image."""
    image = image.astype("uint8")
    # Convert RGB → Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE
    enhanced = clahe.apply(gray)
    # Convert back to RGB (3-channel)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced


# ─── Single Image Preprocessing ───────────────────────────────────────────────
def preprocess_image(image):
    """Resize → CLAHE enhance → normalize a single MRI image."""
    # Resize to standard CNN input size
    image = cv2.resize(image, IMG_SIZE)
    # Apply CLAHE enhancement
    image = enhance_image(image)
    # Normalize pixel values [0-255] → [0.0-1.0]
    image = image.astype(np.float32) / 255.0
    return image


# ─── Batch Preprocessing (tf.data compatible) ─────────────────────────────────
def preprocess_batch(images, labels):
    """Map preprocess_image over a batch using tf.numpy_function."""
    processed_images = tf.map_fn(
        lambda image: tf.numpy_function(
            preprocess_image,
            [image],
            tf.float32
        ),
        images,
        fn_output_signature=tf.float32
    )
    processed_images.set_shape([None, IMG_SIZE[0], IMG_SIZE[1], 3])
    return processed_images, labels


# ─── Dataset Loader ───────────────────────────────────────────────────────────
def load_dataset(directory, shuffle=True):
    """
    Load images from directory into a tf.data.Dataset.
    Expects sub-folders named after each class.
    Returns: (dataset, class_names)
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        label_mode="int",
        class_names=CLASSES,
        seed=42
    )
    return dataset


# ─── Build Final Preprocessed Datasets ───────────────────────────────────────
def build_datasets():
    """
    Load, preprocess, and prefetch train/test datasets.
    Returns: (train_processed, test_processed)
    """
    train_dataset = load_dataset(TRAIN_DIR, shuffle=True)
    test_dataset  = load_dataset(TEST_DIR,  shuffle=False)

    train_processed = train_dataset.map(
        preprocess_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    test_processed = test_dataset.map(
        preprocess_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_processed, test_processed


# ─── Entry point (for testing pipeline standalone) ───────────────────────────
if __name__ == "__main__":
    print("Loading and preprocessing datasets...")
    train_ds, test_ds = build_datasets()
    for images, labels in train_ds.take(1):
        print(f"  Batch shape : {images.shape}")
        print(f"  Labels      : {labels.numpy()}")
        print(f"  Pixel range : [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
    print("Preprocessing pipeline OK.")
