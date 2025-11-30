from tensorflow.keras import layers, models, Input

def build_model(vocab_size):

    inputs = Input(shape = (128, 256, 1)) #Images 128 pix high, 256 pix wide, and 1 since greyscale

    # CNN encoder (converts image into a sequence of features)

    x = layers.Conv2D(32, 3, padding = 'same', activation = 'relu')(inputs)
    x = layers.MaxPooling2D(2)(x) # Maxpooling to avoid overtraining (does random errors so it would be ready for new data)

    x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # shape → (batch, seq_len(height*width), features)
    layers.Reshape((x.shape[1] * x.shape[2], x.shape[3])) #Flatten 1st and 2nd dimensions

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

    outputs = layers.TimeDistributed(
        layers.Dense(vocab_size, activation = 'softmax')
    )
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics =['accuracy'])

    return model



