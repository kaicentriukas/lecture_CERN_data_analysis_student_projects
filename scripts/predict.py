# predict.py with optional beam search
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import argparse
from tokenizer import sequence_to_text, BOS_TOKEN, EOS_TOKEN
from inference_model import build_inference_models
from dataset import preprocess_image  # your image preprocessing
import re
from config import MAX_SEQ_LEN

# -----------------------------
# Step-by-step greedy decoding function
# -----------------------------
def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def greedy_decode(encoder_states, decoder_model, start_id, end_id, max_len=150):
    h1, c1, h2, c2 = encoder_states
    decoded_ids = [start_id]
    for _ in range(max_len):
        token_input = np.array([[decoded_ids[-1]]])
        logits, h1, c1, h2, c2 = decoder_model.predict([token_input, h1, c1, h2, c2], verbose=0)
        logits_step = logits[0, 0] if logits.ndim == 3 else logits[0]
        next_id = int(np.argmax(logits_step))
        decoded_ids.append(next_id)
        if next_id == end_id:
            break
    return decoded_ids

def beam_decode(encoder_states, decoder_model, start_id, end_id, max_len=150, beam_size=5, length_norm=0.7, eos_bonus=0.2, repeat_penalty=0.15):
    h1, c1, h2, c2 = encoder_states
    beams = [{
        'seq': [start_id],
        'logp': 0.0,
        'states': (h1, c1, h2, c2),
        'finished': False
    }]
    finished = []
    for _ in range(max_len):
        new_beams = []
        for b in beams:
            if b['finished']:
                new_beams.append(b)
                continue
            last_id = b['seq'][-1]
            token_input = np.array([[last_id]])
            logits, nh1, nc1, nh2, nc2 = decoder_model.predict([token_input, b['states'][0], b['states'][1], b['states'][2], b['states'][3]], verbose=0)
            logits_step = logits[0, 0] if logits.ndim == 3 else logits[0]
            probs = softmax(logits_step)
            top_idx = np.argpartition(probs, -beam_size)[-beam_size:]
            top_idx = top_idx[np.argsort(-probs[top_idx])]
            for idx in top_idx:
                seq = b['seq'] + [int(idx)]
                # apply repeat penalty for immediate repeats and simple bi-gram loops
                rp = 0.0
                if len(seq) >= 2 and seq[-1] == seq[-2]:
                    rp += repeat_penalty
                if len(seq) >= 3 and seq[-1] == seq[-3]:
                    rp += repeat_penalty * 0.5
                nb = {
                    'seq': seq,
                    'logp': b['logp'] + float(np.log(probs[int(idx)] + 1e-12)) - rp,
                    'states': (nh1, nc1, nh2, nc2),
                    'finished': int(idx) == end_id
                }
                new_beams.append(nb)
        def score(b):
            L = max(1, len(b['seq']) - 1)
            bonus = eos_bonus if b['finished'] else 0.0
            # simple brace balance penalty: count of '{' vs '}' tokens if available in tokenizer
            brace_pen = 0.0
            try:
                open_id = tokenizer.word_index.get('{')
                close_id = tokenizer.word_index.get('}')
                if open_id and close_id:
                    opens = sum(1 for t in b['seq'] if t == open_id)
                    closes = sum(1 for t in b['seq'] if t == close_id)
                    if opens > closes:
                        brace_pen += 0.05 * (opens - closes)
            except Exception:
                pass
            return (b['logp'] + bonus - brace_pen) / (L ** length_norm)
        new_beams.sort(key=score, reverse=True)
        beams = new_beams[:beam_size]
        if all(b['finished'] for b in beams):
            finished.extend(beams)
            break
    candidates = finished if finished else beams
    best = max(candidates, key=lambda b: (b['logp'] + (eos_bonus if b['finished'] else 0.0)) / (max(1, len(b['seq']) - 1) ** length_norm))
    return best['seq']

# -----------------------------
# Load trained model
# -----------------------------
def load_latest_model():
    print("[INFO] Loading model...")
    try:
        m = load_model("best_model.h5", compile=False)
        print("[INFO] Using model: best_model.h5")
        return m
    except Exception:
        m = load_model("model_gpu.h5", compile=False)
        print("[INFO] Using model: model_gpu.h5")
        return m

# -----------------------------
# Build inference encoder/decoder
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default="../dataset/hand_data/prediction_test_4.jpg", help='Path to image file')
parser.add_argument('--beam', type=int, default=1, help='Beam size (1=greedy)')
parser.add_argument('--length-norm', type=float, default=0.7, help='Length normalization exponent')
parser.add_argument('--model', type=str, default=None, help='Explicit model path to use (overrides best_model.h5/model_gpu.h5)')
args = parser.parse_args()

if args.model:
    print(f"[INFO] Loading model override: {args.model}")
    training_model = load_model(args.model, compile=False)
    print(f"[INFO] Using model: {args.model}")
else:
    training_model = load_latest_model()
print("[INFO] Building inference models...")
encoder_model, decoder_model = build_inference_models(training_model)

print("[INFO] Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("[INFO] Tokenizer loaded. vocab_size=", getattr(tokenizer, "num_words", None) or len(getattr(tokenizer, "word_index", {})))

# Safe retrieval of BOS/EOS ids
start_id = tokenizer.word_index.get(BOS_TOKEN, 1)
end_id = tokenizer.word_index.get(EOS_TOKEN, 2)

print(f"[INFO] Preprocessing image: {args.image}")
img = preprocess_image(args.image)
img = np.expand_dims(img, axis=0)

# -----------------------------
# Encoder forward pass
# -----------------------------
h1, c1 = encoder_model.predict(img)
h2, c2 = np.zeros_like(h1), np.zeros_like(c1)

# -----------------------------
# Run greedy decoding
# -----------------------------
if args.beam > 1:
    decoded_ids = beam_decode(
        encoder_states=(h1, c1, h2, c2),
        decoder_model=decoder_model,
        start_id=start_id,
        end_id=end_id,
        max_len=MAX_SEQ_LEN,
        beam_size=args.beam,
        length_norm=args.length_norm,
    )
else:
    decoded_ids = greedy_decode(
        encoder_states=(h1, c1, h2, c2),
        decoder_model=decoder_model,
        start_id=start_id,
        end_id=end_id,
        max_len=MAX_SEQ_LEN
    )

# -----------------------------
# Convert token IDs to LaTeX
# -----------------------------
decoded_ids_no_bos = decoded_ids[1:]
latex_text = sequence_to_text(tokenizer, decoded_ids_no_bos)

latex_text_clean = re.sub(r'(.)\1{1,}', r'\1', latex_text)

# Remove any residual BOS/EOS tokens
latex_text_clean = latex_text_clean.replace(BOS_TOKEN, "").replace(EOS_TOKEN, "")

print("Decoded LaTeX:", latex_text_clean)
