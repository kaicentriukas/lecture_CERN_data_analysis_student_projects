# predict_debug.py
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from tokenizer import sequence_to_text, BOS_TOKEN, EOS_TOKEN
from inference_model import build_inference_models
from dataset import preprocess_image  # your image preprocessing

# -----------------------------
# Load trained model
# -----------------------------
print("[INFO] Loading model...")
training_model = load_model("model_gpu.h5", compile=False)

# -----------------------------
# Build inference encoder/decoder
# -----------------------------
print("[INFO] Building inference models...")
encoder_model, decoder_model = build_inference_models(training_model)

# -----------------------------
# Load tokenizer
# -----------------------------
print("[INFO] Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Safe retrieval of BOS/EOS ids
start_id = tokenizer.word_index.get(BOS_TOKEN)
if start_id is None:
    print(f"[WARN] BOS token {BOS_TOKEN} not found in tokenizer. Using 1 as default.")
    start_id = 1

end_id = tokenizer.word_index.get(EOS_TOKEN)
if end_id is None:
    print(f"[WARN] EOS token {EOS_TOKEN} not found in tokenizer. Using 2 as default.")
    end_id = 2

# -----------------------------
# Load & preprocess image
# -----------------------------
img_path = "../dataset/hand_data/prediction_test_2.jpg"
print(f"[INFO] Preprocessing image: {img_path}")
img = preprocess_image(img_path)

print(f"[DEBUG] Preprocessed image shape (before batch): {img.shape}, min={img.min()}, max={img.max()}")
img = np.expand_dims(img, axis=0)
print(f"[DEBUG] Image shape after adding batch dimension: {img.shape}")

# -----------------------------
# Encoder forward pass
# -----------------------------
print("[INFO] Running encoder forward pass...")
h1, c1 = encoder_model.predict(img)
h2 = np.zeros_like(h1)
c2 = np.zeros_like(c1)
print(f"[DEBUG] Encoder states shapes: h1={h1.shape}, c1={c1.shape}, h2={h2.shape}, c2={c2.shape}")

# -----------------------------
# Step-by-step greedy decoding
# -----------------------------
max_len = 150
decoded_ids = [start_id]

print("[INFO] Starting greedy decoding loop...")
for i in range(max_len):
    token_input = np.array([[decoded_ids[-1]]])
    logits, h1, c1, h2, c2 = decoder_model.predict([token_input, h1, c1, h2, c2])
    next_id = int(np.argmax(logits[0, 0]))
    decoded_ids.append(next_id)
    
    # Debug info
    print(f"[STEP {i+1}] last token ID: {decoded_ids[-2]}, next ID: {next_id}")
    
    if next_id == end_id:
        print(f"[INFO] EOS token reached at step {i+1}")
        break

# -----------------------------
# Convert token IDs to LaTeX
# -----------------------------
decoded_ids_no_bos = decoded_ids[1:]
latex_text = sequence_to_text(tokenizer, decoded_ids_no_bos)
print("[DEBUG] Raw decoded text (with possible extra BOS/EOS):", latex_text)

# Clean up any residual BOS/EOS tokens
latex_text_clean = latex_text.replace(BOS_TOKEN, "").replace(EOS_TOKEN, "")
print("[INFO] Predicted LaTeX:", latex_text_clean)
