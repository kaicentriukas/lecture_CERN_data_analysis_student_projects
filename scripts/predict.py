# predict_fullseq.py
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import os
from dataset import preprocess_image
from tokenizer import sequence_to_text
from config import MAX_SEQ_LEN

MODEL_PATH = "model_gpu.h5"
TOKENIZER_PATH = "tokenizer.pkl"
IMAGE_PATH = "../dataset/hand_data/prediction_test_1.jpg"

# -------------------
# Helpers
# -------------------
def prepare_image_for_model(image_path, target_size=(64, 128)):
    # Open file, preprocess using your preprocess_image which expects PIL.Image or np.array
    img = Image.open(image_path).convert("L")
    # make sure preprocess_image uses same IMG_SIZE as training; if not, change target_size
    img = img.resize((target_size[1], target_size[0]))
    # Convert to numpy float32 0..1 and add channel
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found.")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"{TOKENIZER_PATH} not found.")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"{IMAGE_PATH} not found.")

    # load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    model = load_model(MODEL_PATH, compile=False)  # compile=False is fine for inference

    img = prepare_image_for_model(IMAGE_PATH, target_size=(64, 128))
    img_batch = np.expand_dims(img, axis=0)  # (1, H, W, 1)

    # Build a dummy decoder input of the same length used at training:
    # If your training used MAX_SEQ_LEN and sequences shaped (max_len,) then:
    decoder_len = MAX_SEQ_LEN - 1
    decoder_in = np.zeros((1, decoder_len), dtype=np.int32)

    # If tokenizer has <START>, insert it; else keep zeros (pad)
    start_id = tokenizer.word_index.get('<START>', 0)
    decoder_in[0, 0] = start_id

    # Predict full sequence at once (model was trained with teacher forcing)
    preds = model.predict([img_batch, decoder_in], verbose=0)  # shape (1, decoder_len, vocab_size)

    # Greedy argmax across time steps:
    token_ids = list(np.argmax(preds[0], axis=-1))  # list length = decoder_len

    # Trim at <END> if present
    end_id = tokenizer.word_index.get('<END>')
    if end_id is not None:
        if end_id in token_ids:
            token_ids = token_ids[:token_ids.index(end_id)]

    # Convert to text; fall back to manual mapping if needed
    latex = sequence_to_text(tokenizer, token_ids).strip()
    if not latex:
        index_word = {v: k for k, v in tokenizer.word_index.items()}
        latex = ''.join(index_word.get(t, '') for t in token_ids if t != 0).strip()

    print("Predicted token ids (first 50):", token_ids[:50])
    print("\nFinal LaTeX prediction:", latex)
