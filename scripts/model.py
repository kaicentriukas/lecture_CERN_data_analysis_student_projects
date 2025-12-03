from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def build_model(vocab_size, output_seq_len=100, learning_rate=0.001):
    """
    Lightweight encoder-decoder model for MathWriting.
    Optimized for speed on GPU.
    """
    # -------------------------------
    # ENCODER: Lightweight CNN
    # -------------------------------
    img_input = Input(shape=(48, 96, 1))  # Match ultra-fast dataset

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Map to initial LSTM states
    init_h = layers.Dense(128, activation='relu')(x)
    init_c = layers.Dense(128, activation='relu')(x)

    # -------------------------------
    # DECODER: CuDNN LSTM (fast on GPU)
    # -------------------------------
    dec_input = Input(shape=(output_seq_len,), dtype='int32')

    emb = layers.Embedding(vocab_size, 96, mask_zero=True)(dec_input)
    lstm_out = layers.LSTM(
        128,
        return_sequences=True,
        dropout=0.0,
        recurrent_dropout=0.0
    )(emb, initial_state=[init_h, init_c])

    logits = layers.Dense(vocab_size, activation='softmax')(lstm_out)

    model = models.Model([img_input, dec_input], logits)

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
