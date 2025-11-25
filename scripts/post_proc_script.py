import cv2
import os
import sys
import cv2

tp = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = cv2.resize(img, (800, 200))
    return img


script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.normpath(os.path.join(script_dir, '..', 'dataset'))
index = 0
filepath = os.path.join(dataset_dir, f"formula_{index:03d}.png")
for i in range(100):
    filepath = os.path.join(dataset_dir, f"formula_{i:03d}.png")
    if not os.path.exists(filepath):
        print(f"Skipping missing image: {filepath}")
        continue
    processed_img = preprocess_image(filepath)
    cv2.imwrite(filepath, processed_img)

