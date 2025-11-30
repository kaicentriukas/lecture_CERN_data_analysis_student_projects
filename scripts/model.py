from tensorflow.keras import layers, models, Input

def build_model(vocab_size, output_seq_len=150):
    inputs = Input(shape=(128, 256, 1))

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Global average pooling and repeat to fixed sequence length
    x = layers.GlobalAveragePooling2D()(x)         # (batch, channels)
    x = layers.RepeatVector(output_seq_len)(x)     # (batch, output_seq_len, channels)

    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(x)

    model = models.Model(inputs, x)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
