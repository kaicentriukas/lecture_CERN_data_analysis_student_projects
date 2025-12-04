from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from model import build_model  # your model.py replacement

def build_inference_models(training_model):
    """
    Creates encoder and decoder models for step-by-step inference.
    Handles two stacked LSTM layers properly.
    """

    # -----------------------------
    # Encoder model
    # -----------------------------
    encoder_input = training_model.get_layer("encoder_input").input
    init_h = training_model.get_layer("init_h").output
    init_c = training_model.get_layer("init_c").output

    encoder_model = Model(inputs=encoder_input, outputs=[init_h, init_c], name="encoder_model")

    # -----------------------------
    # Decoder model
    # -----------------------------
    decoder_input = training_model.get_layer("decoder_input").input
    dec_emb_layer = training_model.get_layer("decoder_embedding")
    dec_ln_layer = training_model.get_layer("decoder_ln")
    lstm1_layer = training_model.get_layer("decoder_lstm1")
    lstm2_layer = training_model.get_layer("decoder_lstm2")
    proj_layer = training_model.get_layer("proj")
    logits_layer = training_model.get_layer("decoder_logits")

    # Step input tokens
    token_in = Input(shape=(1,), dtype='int32', name="token_in")
    # Previous LSTM states
    h1_in = Input(shape=(128,), name="h1_in")
    c1_in = Input(shape=(128,), name="c1_in")
    h2_in = Input(shape=(128,), name="h2_in")
    c2_in = Input(shape=(128,), name="c2_in")

    # Embedding + LayerNorm
    emb = dec_emb_layer(token_in)
    emb = dec_ln_layer(emb)

    # LSTM layers
    lstm1_out, h1_out, c1_out = lstm1_layer(emb, initial_state=[h1_in, c1_in])
    lstm2_out, h2_out, c2_out = lstm2_layer(lstm1_out, initial_state=[h2_in, c2_in])

    # Projection + logits
    proj = proj_layer(lstm2_out)
    logits = logits_layer(proj)

    decoder_model = Model(
        inputs=[token_in, h1_in, c1_in, h2_in, c2_in],
        outputs=[logits, h1_out, c1_out, h2_out, c2_out],
        name="decoder_model"
    )

    return encoder_model, decoder_model
