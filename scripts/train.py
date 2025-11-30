from model import build_model
from tokenizer import create_char_tokenizer, texts_to_sequences
from dataset import load_mathwriting, preprocess_image, create_tf_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load data

ds = load_mathwriting()

# Extract formulas(LaTex labels) and images
latex_texts = []
images = []

for item in ds['train'][:20000]: #right now limiting the amount data loaded
    latex_texts.append(item['latex'])
    images.append(item['image'])

# Tokenizer (assign text to numbers)
tokenizer = create_char_tokenizer(latex_texts)

# Convert text formulas into integer sequences
label_sequences = texts_to_sequences(tokenizer, latex_texts, max_len = 150)

vocab_size = len(tokenizer.word_index) + 1

# Preprocess the images 

processed_images = [] #For storing tensor images

# Converting from PIL to model ready (greyscale and normalized)

for img in images:
    ready_img = preprocess_image(img)
    processed_images.append(ready_img)

# Creating the tensorflow Dataset
train_dataset = create_tf_dataset(processed_images, label_sequences, batch_size = 32)

# Build the model

model = build_model(vocab_size)
model.summary()

# Training loop
epochs = 5
history = model.fit(train_dataset, epochs)

# Save trained model
model.save('model.h5')


# -------------------------------------------------------
# PLOT TRAINING LOSS
# -------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"])
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# -------------------------------------------------------
# PLOT TRAINING ACCURACY (IF your model has accuracy)
# -------------------------------------------------------
if "accuracy" in history.history:
    plt.figure(figsize=(8,5))
    plt.plot(history.history["accuracy"])
    plt.title("Training Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
else:
    print("⚠️ Your model has no 'accuracy' metric — loss plot only.")
