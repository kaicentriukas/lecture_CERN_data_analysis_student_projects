# dataset.py (ULTRA FAST VERSION)
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset

# Reduce image size for speed
IMG_SIZE = (48, 96)   # (height, width) → 4× faster CNN


# ---------------------------------------------------------
# LOAD HUGGINGFACE DATASET
# ---------------------------------------------------------
def load_mathwriting(limit=None):
    """
    Loads the MathWriting dataset and returns:
    (train_images, train_latex), (val_images, val_latex)
    """

    ds = load_dataset("deepcopy/MathWriting-human")

    train_imgs, train_latex = [], []
    val_imgs, val_latex = [], []

    for sample in ds["train"]:
        if limit is not None:
            if len(train_imgs) + len(val_imgs) >= limit:
                break

        img = sample["image"]
        tex = sample["latex"]
        tag = sample["split_tag"]

        if tag == "train":
            train_imgs.append(img)
            train_latex.append(tex)
        elif tag == "val":
            val_imgs.append(img)
            val_latex.append(tex)

    return (train_imgs, train_latex), (val_imgs, val_latex)


# ---------------------------------------------------------
# FAST IMAGE PREPROCESSING (PIL → NumPy)
# ---------------------------------------------------------
def preprocess_image(img):
    """
    Convert PIL or NumPy image to a normalized 48x96 grayscale float tensor.
    """

    # Convert to numpy
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise ValueError(f"Unknown image type: {type(img)}")

    # Convert to grayscale
    img = img.convert("L")

    # Resize to (width, height)
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))

    # Convert to float32 tensor (0–1)
    img = np.array(img, dtype=np.float32) / 255.0

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)

    return img


# ---------------------------------------------------------
# EXTREMELY FAST TF.DATA CREATION
# ---------------------------------------------------------
def create_tf_dataset(images, sequences, batch_size=32):
    """
    This version converts *everything* to TensorFlow tensors first,
    then slices without Python overhead. 10–20× faster.
    """

    # Convert to TF tensors (MUCH faster than feeding Python lists)
    images = tf.constant(np.array(images), dtype=tf.float32)       # (N, 48, 96, 1)
    sequences = tf.constant(np.array(sequences), dtype=tf.int32)   # (N, max_len)

    # Prepare decoder input/output (shifted by 1)
    decoder_in  = sequences[:, :-1]
    decoder_out = sequences[:, 1:]

    # Build dataset
    ds = tf.data.Dataset.from_tensor_slices(
        ((images, decoder_in), decoder_out)
    )

    # Shuffle → batch → prefetch
    ds = (
        ds.shuffle(512)            # Large shuffle buffer = better randomization
          .batch(batch_size)        # Efficient batching
          .prefetch(tf.data.AUTOTUNE)
    )

    return ds
