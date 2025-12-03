from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from config import MAX_SEQ_LEN

def build_model(vocab_size, output_seq_len=MAX_SEQ_LEN, learning_rate=0.001):
    """
    Lightweight encoder-decoder model for MathWriting.
    Optimized for speed on GPU.
    """
    # -------------------------------
    # ENCODER: Lightweight CNN
    # -------------------------------
    img_input = Input(shape=(64, 128, 1))  # Match ultra-fast dataset

    x = layers.Conv2D(48, 3, padding='same', activation='relu')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(96, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(160, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(320, activation='relu')(x)
        # Map to initial LSTM states
    init_h = layers.Dense(128, activation='relu')(x)
    init_c = layers.Dense(128, activation='relu')(x)

    # -------------------------------
    # DECODER: CuDNN LSTM (fast on GPU)
    # -------------------------------
    pos_emb = layers.Embedding(output_seq_len, 96)
    positions = tf.range(start=0, limit=output_seq_len, delta=1)
    positional_encoding = pos_emb(positions)

    dec_input = Input(shape=(None,), dtype='int32')

    emb = layers.Embedding(vocab_size, 96, mask_zero=True)(dec_input)
    emb = layers.Dropout(0.1)(emb)
    emb = emb + positional_encoding[:tf.shape(emb)[1]]
    emb = layers.LayerNormalization()(emb)

    lstm_out = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.1,
            recurrent_dropout=0.0
        )
    )(emb, initial_state=[init_c, init_h] * 2)  # bidirectional needs h and c for both directions

    proj = layers.Dense(128, activation='relu')(lstm_out)
    logits = layers.Dense(vocab_size, activation='softmax')(proj)


    model = models.Model([img_input, dec_input], logits)

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model
