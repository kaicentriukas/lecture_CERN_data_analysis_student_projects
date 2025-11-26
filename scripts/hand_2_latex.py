import numpy as np
from datasets import load_dataset

ds = load_dataset("deepcopy/MathWriting-human")

def generator():
    for item in ds["train"]:
        img = np.array(item["image"]) / 255.0
        latex = item["latex"]
        yield img, latex  # (for CNN you'll encode latex separately)
