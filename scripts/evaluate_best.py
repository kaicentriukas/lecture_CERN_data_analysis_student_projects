import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset import load_mathwriting, preprocess_image, create_tf_dataset
from tokenizer import create_char_tokenizer, texts_to_sequences
from config import MAX_SEQ_LEN, DATA_LIMIT, BATCH_SIZE, SEED

# Deterministic seeds
# Use shared seed
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load best model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"best_model.h5 not found at {MODEL_PATH}. Run training first.")

model = load_model(MODEL_PATH)

# Load a consistent validation split
(_, _), (val_images, val_latex) = load_mathwriting(limit=DATA_LIMIT)
if len(val_images) == 0:
    # Fallback: take last 10% as validation deterministically
    from sklearn.model_selection import train_test_split
    train_images, val_images, train_latex, val_latex = train_test_split(
        val_images, val_latex, test_size=0.1, random_state=SEED
    )

# Tokenize using same max length as training default
# If a tokenizer.pkl exists, use it for consistent vocab
TOK_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")
if os.path.exists(TOK_PATH):
    with open(TOK_PATH, "rb") as f:
        tokenizer = pickle.load(f)
else:
    tokenizer = create_char_tokenizer(val_latex)

val_sequences = texts_to_sequences(tokenizer, val_latex, max_len=MAX_SEQ_LEN)

# Preprocess images once
processed_val_images = [preprocess_image(img) for img in val_images]

# Build eval dataset (no augmentation)
# Use shared batch size
val_dataset = create_tf_dataset(processed_val_images, val_sequences, batch_size=BATCH_SIZE)

# Evaluate
results = model.evaluate(val_dataset, verbose=1)
print({"loss": float(results[0]), "accuracy": float(results[1])})
