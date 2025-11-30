# model.py
from tensorflow.keras import layers, models, Input

def build_model(vocab_size, output_seq_len=150):
    # Encoder
    img_input = Input(shape=(128, 256, 1))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(img_input)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Project to initial LSTM states
    init_h = layers.Dense(256, activation='tanh')(x)
    init_c = layers.Dense(256, activation='tanh')(x)

    # Decoder with teacher forcing
    dec_input = Input(shape=(output_seq_len,), dtype='int32')
    emb = layers.Embedding(vocab_size, 128, mask_zero=True)(dec_input)
    lstm_out = layers.LSTM(256, return_sequences=True)(emb, initial_state=[init_h, init_c])
    lstm_out = layers.LSTM(256, return_sequences=True)(lstm_out)
    logits = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(lstm_out)

    model = models.Model([img_input, dec_input], logits)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
