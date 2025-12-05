# dataset.py (ULTRA FAST VERSION)
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from datasets import load_dataset

# Image size (height, width). Increased to capture more detail.
# Note: Keep within GPU memory limits (~2GB). Adjust if OOM.
IMG_SIZE = (80, 160)


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
    Convert path/PIL/NumPy image to normalized grayscale float tensor of size IMG_SIZE.
    Accepts:
    - str: filesystem path to an image
    - PIL.Image.Image
    - np.ndarray (uint8 or float)
    """

    # Load if a file path string is provided
    if isinstance(img, str):
        img = Image.open(img)

    # Convert ndarray to PIL
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

    # Validate type
    if not isinstance(img, Image.Image):
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
def create_tf_dataset(images, sequences, batch_size=32, augment=False):
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


    # Shuffle → batch → prefetch (always batch, even without augmentation)
    ds = (
        ds.cache()
            .shuffle(512)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
    )

    return ds
