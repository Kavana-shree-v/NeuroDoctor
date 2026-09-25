"""
Brain Tumor MRI — CNN Model (Customized_CNN_V5)
Architecture: SE Residual Attention Blocks + Dense head
Classes: glioma | meningioma | notumor | pituitary
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from preprocessing import build_datasets, CLASSES

# ─── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES  = 4
IMG_SIZE     = (224, 224, 3)
MODEL_PATH   = os.path.join("model", "brain_tumor_model.keras")
os.makedirs("model", exist_ok=True)


# ─── Squeeze-and-Excitation Block ─────────────────────────────────────────────
def squeeze_excitation_block(x, filters, ratio=8):
    """
    Channel attention: squeeze global info → excite channel weights.
    ratio: reduction ratio for the bottleneck FC layers.
    """
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(max(filters // ratio, 1), activation="relu",  use_bias=False)(se)
    se = layers.Dense(filters,                  activation="sigmoid", use_bias=False)(se)
    return layers.Multiply()([x, se])


# ─── Residual Attention Block (SE + skip connection + dropout) ────────────────
def residual_attention_block(x, filters, dropout_rate=0.10):
    """
    Conv → BN → ReLU → Conv → BN → SE → Add(skip) → ReLU → Dropout
    Skip connection uses 1×1 Conv projection when channel count changes.
    """
    shortcut = x

    # Main path
    out = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)

    out = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)

    # SE channel attention
    out = squeeze_excitation_block(out, filters)

    # Projection shortcut if channels differ
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Residual add + activation
    out = layers.Add()([out, shortcut])
    out = layers.Activation("relu")(out)
    out = layers.Dropout(dropout_rate)(out)
    return out


# ─── Build Model V5 ───────────────────────────────────────────────────────────
def build_model():
    """
    Customized_CNN_V5:
      Stem Conv → 4 × SE-Residual stages (32/64/128/256 filters)
      → GAP → Dense(128) → Dense(64) → Softmax(4)
    """
    inputs = layers.Input(shape=IMG_SIZE)

    # Stem
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Stage 1 — 32 filters
    x = residual_attention_block(x, 32,  dropout_rate=0.10)
    x = layers.MaxPooling2D((2, 2))(x)

    # Stage 2 — 64 filters
    x = residual_attention_block(x, 64,  dropout_rate=0.12)
    x = layers.MaxPooling2D((2, 2))(x)

    # Stage 3 — 128 filters
    x = residual_attention_block(x, 128, dropout_rate=0.15)
    x = layers.MaxPooling2D((2, 2))(x)

    # Stage 4 — 256 filters (no pooling — feeds directly into GAP)
    x = residual_attention_block(x, 256, dropout_rate=0.18)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64,  activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Customized_CNN_V5")
    return model


# ─── Compile ──────────────────────────────────────────────────────────────────
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",   # dataset uses int labels
        metrics=["accuracy"]
    )
    return model


# ─── Callbacks ────────────────────────────────────────────────────────────────
def get_callbacks():
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1
        ),
    ]


# ─── Evaluate on Test Set ─────────────────────────────────────────────────────
def evaluate_model(model, test_ds):
    """Run predictions on the test set and print classification report."""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix — Customized_CNN_V5")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Confusion matrix saved → results/confusion_matrix.png")

    return y_true, y_pred


# ─── Plot Training History ────────────────────────────────────────────────────
def plot_history(history):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "training_history.png"), dpi=150)
    plt.close()
    print("Training history saved → results/training_history.png")


# ─── Train ────────────────────────────────────────────────────────────────────
def train():
    print("Building datasets...")
    train_ds, test_ds = build_datasets()

    # Use 10% of training data for validation
    total_batches   = tf.data.experimental.cardinality(train_ds).numpy()
    val_batches     = max(1, int(total_batches * 0.10))
    val_ds          = train_ds.take(val_batches)
    training_ds     = train_ds.skip(val_batches)

    print("Building model...")
    model = build_model()
    model = compile_model(model)
    model.summary()

    print("\nStarting training...")
    history = model.fit(
        training_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=get_callbacks()
    )

    plot_history(history)

    print("\nEvaluating on test set...")
    evaluate_model(model, test_ds)

    print(f"\nBest model saved → {MODEL_PATH}")
    return model, history


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
