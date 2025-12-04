from model import build_model
from tokenizer import create_char_tokenizer, texts_to_sequences
from dataset import load_mathwriting, preprocess_image, create_tf_dataset
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy
from config import MAX_SEQ_LEN, BATCH_SIZE, DATA_LIMIT, SEED
import os
import random
import numpy as np


# -------------------------------
# Mixed precision / GPU check
# -------------------------------
set_global_policy("float32")  # TF 2.10 mixed precision safe with float32

# -------------------------------
# Deterministic seeds
# -------------------------------
# Seed from shared config
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if len(gpus) == 0:
    print("WARNING: No GPU detected. Training will run on CPU.")

# -------------------------------
# Load dataset (limit for speed)
# -------------------------------
(train_images, train_latex), (val_images, val_latex) = load_mathwriting(limit=DATA_LIMIT)

# If no val split, create one
if len(val_images) == 0:
    from sklearn.model_selection import train_test_split
    train_images, val_images, train_latex, val_latex = train_test_split(
        train_images, train_latex, test_size=0.1, random_state=SEED
    )

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = create_char_tokenizer(train_latex)
train_sequences = texts_to_sequences(tokenizer, train_latex, max_len=MAX_SEQ_LEN)
val_sequences = texts_to_sequences(tokenizer, val_latex, max_len=MAX_SEQ_LEN)
vocab_size = len(tokenizer.word_index) + 1  # include padding index

# -------------------------------
# Preprocess images ONCE (CPU)
# -------------------------------
processed_train_images = [preprocess_image(img) for img in train_images]
processed_val_images = [preprocess_image(img) for img in val_images]

# -------------------------------
# Create ultra-fast TF datasets
# -------------------------------
# Batch size from shared config
train_dataset = create_tf_dataset(processed_train_images, train_sequences, batch_size=BATCH_SIZE)
val_dataset   = create_tf_dataset(processed_val_images, val_sequences, batch_size=BATCH_SIZE)

# -------------------------------
# Build model
# -------------------------------
model = build_model(vocab_size, output_seq_len=MAX_SEQ_LEN-1)  # shift for decoder
model.summary()

# -------------------------------
# Callbacks
# -------------------------------
checkpoint_cb = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

# -------------------------------
# Train
# -------------------------------
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, reduce_lr, early_stop],
    verbose=1
)

# -------------------------------
# Save model & tokenizer
# -------------------------------
model.save("model_gpu.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training finished successfully!")
