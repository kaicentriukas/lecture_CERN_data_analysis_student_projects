from model import build_model
from tokenizer import create_char_tokenizer, texts_to_sequences
from dataset import load_mathwriting, preprocess_image, create_tf_dataset
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy("float32")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# -------------------------------
# Load dataset
# -------------------------------
(train_images, train_latex), (val_images, val_latex) = load_mathwriting(limit=5000)
if len(val_images) == 0:
    from sklearn.model_selection import train_test_split
    train_images, val_images, train_latex, val_latex = train_test_split(
        train_images, train_latex, test_size=0.1, random_state=42
    )

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = create_char_tokenizer(train_latex)
train_sequences = texts_to_sequences(tokenizer, train_latex, max_len=100)
val_sequences = texts_to_sequences(tokenizer, val_latex, max_len=100)
vocab_size = len(tokenizer.word_index) + 1

# -------------------------------
# Preprocess images ONCE
# -------------------------------
processed_train_images = [preprocess_image(img) for img in train_images]
processed_val_images = [preprocess_image(img) for img in val_images]

# -------------------------------
# Create ultra-fast TF datasets
# -------------------------------
batch_size = 128  # GPU can handle larger batches
train_dataset = create_tf_dataset(processed_train_images, train_sequences, batch_size=batch_size)
val_dataset   = create_tf_dataset(processed_val_images, val_sequences, batch_size=batch_size)

# -------------------------------
# Build model
# -------------------------------
model = build_model(vocab_size, output_seq_len=99)  # 100 tokens minus 1 for decoder shift
model.summary()

# -------------------------------
# Training callbacks
# -------------------------------
checkpoint_cb = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

# -------------------------------
# Train
# -------------------------------
steps = len(processed_train_images) // batch_size
validation_steps = len(processed_val_images) // batch_size

history = model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=steps,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[checkpoint_cb, reduce_lr, early_stop],
    verbose=1
)

# -------------------------------
# Save model & tokenizer
# -------------------------------
model.save("model_final.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
