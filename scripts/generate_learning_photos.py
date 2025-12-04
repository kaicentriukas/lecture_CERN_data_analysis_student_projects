from tensorflow.keras.models import load_model
m = load_model("model_gpu.h5", compile=False)
m.summary()
