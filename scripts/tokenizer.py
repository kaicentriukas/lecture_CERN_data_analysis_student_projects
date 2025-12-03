# tokenizer.py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

def create_char_tokenizer(latex_texts, num_chars=5000):
    """
    Character-level tokenizer with BOS/EOS tokens added.
    Much faster and produces better decoding stability.
    """
    # Add start/end tokens to each formula BEFORE training tokenizer
    enhanced_texts = [BOS_TOKEN + text + EOS_TOKEN for text in latex_texts]

    tok = Tokenizer(
        num_words=num_chars,
        filters="", 
        lower=False,
        char_level=True
    )
    tok.fit_on_texts(enhanced_texts)

    return tok


def texts_to_sequences(tok, texts, max_len=150):
    """
    Extremely optimized: no Python loops, no nested lists.
    Adds BOS/EOS automatically.
    """
    enhanced = [BOS_TOKEN + t + EOS_TOKEN for t in texts]

    # This is vectorized internally by TF/Keras → VERY FAST
    sequences = tok.texts_to_sequences(enhanced)

    # Pad to uniform length
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )

    return padded.astype(np.int32)


def sequence_to_text(tok, seq):
    """
    Converts a sequence of integers back into text, removing padding/BOS/EOS.
    """
    index_word = tok.index_word

    chars = []
    for idx in seq:
        if idx == 0:
            continue
        ch = index_word.get(idx, "")
        if ch in (BOS_TOKEN, EOS_TOKEN):
            continue
        chars.append(ch)

    return "".join(chars)
