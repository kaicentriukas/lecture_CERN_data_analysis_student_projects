from pix2tex.cli import LatexOCR
from PIL import Image
import os 
import cv2
model = LatexOCR()

dataset_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'math_pngs'))
for i in range(100):
    filepath = os.path.join(dataset_dir, f"formula_{i:03d}.png")
    if not os.path.exists(filepath):
        print(f"Skipping missing image: {filepath}")
        continue
    image = Image.open(filepath)
    latex = model(image)
    print( latex)