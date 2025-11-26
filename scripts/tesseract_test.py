from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def trocr_predict(img_path):
    # Ensure img_path is a string path
    if not isinstance(img_path, str):
        raise TypeError(f"Expected img_path to be a string path, got {type(img_path)}")

    # Load image using PIL
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {img_path}")
        raise e

    print("Loaded image size:", image.size)

    # Convert image → pixel values correctly
    enc = processor(image, return_tensors="pt")
    pixel_values = enc.pixel_values

    # Generate output tokens
    generated_ids = model.generate(pixel_values)

    # Decode tokens → text
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("OCR output:", repr(text))

    return text


# ---- Test call ----
img_path = "../dataset/formula_000.png"
trocr_predict(img_path)
