"""
Brain Tumor MRI — Grad-CAM XAI
Generates heatmap overlays showing which regions drove the model's prediction.
"""

import numpy as np
import cv2
import tensorflow as tf


# ─── Find the last Conv2D layer name ──────────────────────────────────────────
def get_last_conv_layer(model):
    """Return the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


# ─── Core Grad-CAM ────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array, class_idx=None, layer_name=None):
    """
    Compute Grad-CAM heatmap for a preprocessed image tensor.

    Args:
        model      : trained Keras model
        img_array  : np.ndarray of shape (1, 224, 224, 3), normalized [0,1]
        class_idx  : int — target class index (None = predicted class)
        layer_name : str — name of Conv2D layer to hook (None = last Conv2D)

    Returns:
        heatmap    : np.ndarray of shape (224, 224), values in [0, 1]
    """
    if layer_name is None:
        layer_name = get_last_conv_layer(model)

    # Build a sub-model: input → [conv_output, final_predictions]
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)          # (1, H, W, C)

    # Global average pooling of gradients → importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]                     # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                      # (H, W)

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_idx


# ─── Overlay Heatmap on Original Image ───────────────────────────────────────
def overlay_gradcam(original_img, heatmap, alpha=0.45):
    """
    Superimpose Grad-CAM heatmap onto the original MRI image.

    Args:
        original_img : np.ndarray (H, W, 3), uint8 [0-255]
        heatmap      : np.ndarray (H, W), float [0-1]
        alpha        : float — heatmap opacity

    Returns:
        superimposed : np.ndarray (H, W, 3), uint8
    """
    # Resize heatmap to original image size
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to colormap (COLORMAP_JET: blue-green-yellow-red)
    heatmap_uint8  = np.uint8(255 * heatmap_resized)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color  = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    superimposed = cv2.addWeighted(
        original_img.astype(np.uint8), 1 - alpha,
        heatmap_color,                 alpha,
        0
    )
    return superimposed


# ─── Full Explain Pipeline ────────────────────────────────────────────────────
def explain(model, preprocessed_img, original_img, layer_name=None):
    """
    Run Grad-CAM and return the overlay image + predicted class index.

    Args:
        model           : trained Keras model
        preprocessed_img: np.ndarray (1, 224, 224, 3), float32 [0,1]
        original_img    : np.ndarray (H, W, 3), uint8  — for overlay
        layer_name      : optional Conv2D layer name override

    Returns:
        overlay     : np.ndarray (H, W, 3), uint8 — heatmap overlay
        heatmap     : np.ndarray (H, W), float    — raw heatmap
        class_idx   : int
    """
    heatmap, class_idx = compute_gradcam(
        model, preprocessed_img, layer_name=layer_name
    )
    overlay = overlay_gradcam(original_img, heatmap)
    return overlay, heatmap, class_idx
