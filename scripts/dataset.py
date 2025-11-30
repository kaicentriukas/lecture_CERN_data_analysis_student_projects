# dataset.py
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset

IMG_SIZE = (128, 256)

def load_mathwriting(limit=20000):
    ds = load_dataset("deepcopy/MathWriting-human")

    subset = ds["train"][:limit]
    images = subset["image"]
    latex = subset["latex"]

    return images, latex


def preprocess_image(img):
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = Image.fromarray(img)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = Image.fromarray(img.squeeze(axis=2))
        elif img.ndim == 3:
            img = Image.fromarray(img)
        else:
            raise ValueError(f"Unexpected shape: {img.shape}")
    elif not isinstance(img, Image.Image):
        raise ValueError(f"Unknown image type: {type(img)}")

    # FORCE correct width x height
    img = img.convert("L").resize((IMG_SIZE[1], IMG_SIZE[0]))  # PIL wants (width, height)

    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


#To not overload memory, we create a tf dataset generator
def create_tf_dataset(images, sequences, batch_size=32):
    def gen():
        for img, seq in zip(images, sequences):
            yield preprocess_image(img), seq #send one image and one latex sequence at a time, to not overload memory

    return tf.data.Dataset.from_generator( 
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32), #tells what size the output will be (128, 256, 1)
            tf.TensorSpec(shape=(None,), dtype=tf.int32), #tells what size the output will be (None for variable length sequences)
        )
    ).padded_batch(batch_size, padded_shapes=([*IMG_SIZE, 1], [None])).prefetch(2)
