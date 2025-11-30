from tensorflow.keras.models import load_model
import tensorflow as tf
from tokenizer import sequence_to_text
from dataset import preprocess_image
import numpy as np
import pickle
import os

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print('Using GPU:', physical_devices)
else:
    print("No GPU found, using CPU")

def beam_search(predictions, beam_width=3):
    """
    Convert model predictions to LaTeX string using beam search.

    Args:
        predictions: np.array of shape (max_len, vocab_size)
        tokenizer: tokenizer object
        beam_width: how many sequences to track at each step

    Returns:
        Predicted LaTeX string
    """
    sequences = [([], 1)] # list of tuples: (sequence, probability)

    # Iterate over each time step
    for t in range(predictions.shape[0]):
        all_candidates = []
        for seq, score in sequences:
            # Get top beam_width probabilities at this step
            top_indices = np.argsort(predictions[t])[-beam_width:]
            for idx in top_indices:
                candidate_seq = seq + [idx] #adds idx value to seq and assigns it to candidate_seq
                candidate_score = score * predictions[t][idx] #Multiply probabilities, where score is probability
                all_candidates.append((candidate_seq, candidate_score))

        # Keeping only the top probabilities
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    best_seq = sequences[0][0] #Select the best one
    return best_seq


def predict_formula(model_path, tokenizer, image_path, beam_width, max_len = 150):
    # Load the training model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)

    # Preprocess image
    img = preprocess_image(image_path) #Shape now (128, 128)
    img = img.reshape((1, img.shape[0], img.shape[1], 1)) #Reshape for batch size + channels (both 1 rn), to get(1, 128, 128, 1)

    # Make predictions oooh
    pred = model.predict(img)

    # pick the best sequence
    best_seq = beam_search(pred, beam_width = beam_width)

    # convert seq of integers into latex strings
    latex = sequence_to_text(tokenizer, best_seq)
    latex = latex.strip()

    return latex

# Example