import pytesseract
import cv2
from PIL import Image
import os
import shutil
import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Auto-detect the Tesseract binary. If it's not found, print clear instructions.
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Example: read one image from the dataset and OCR it
# Compute dataset path relative to this script file so it works regardless of CWD

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.normpath(os.path.join(script_dir, '..', 'dataset'))
index = 12
filepath = os.path.join(dataset_dir, f"formula_{index:03d}.png")
print(f"Resolved image path: {filepath}")
if not os.path.exists(filepath):
	print(f"Image not found at resolved path: {filepath}")
	sys.exit(1)

image = cv2.imread(filepath)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def trocr_predict(img_path):
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

print(trocr_predict(filepath))
