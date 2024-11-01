from utils import IMAGE_QUERIES, RAW_IMAGE_DIR, PROCESSED_IMAGE_DIR, NUM_PER_QUERY, IMAGE_SIZE
from PIL import Image
import numpy as np
import os
import easyocr

def center_crop_image(image_path: str, size: int=IMAGE_SIZE) -> Image.Image:
    image = Image.open(image_path)
    width, height = image.size
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return image.crop((left, top, right, bottom))

def main(): 
    reader = easyocr.Reader(['en'])

    processed_image_idx = 1
    for query in IMAGE_QUERIES:
        for i in range(1, NUM_PER_QUERY + 1):
            image_path = os.path.join(RAW_IMAGE_DIR, query, f"Image_{i}.jpg")
            try:
                resized_img = center_crop_image(image_path, IMAGE_SIZE)
                resized_img_np = np.array(resized_img)
                detected_text = reader.readtext(resized_img_np)
                if len(detected_text) == 0:
                    continue
                output_path = os.path.join(PROCESSED_IMAGE_DIR, f"Image_{processed_image_idx}.webp")
                processed_image_idx += 1
                resized_img.save(output_path, format="WEBP", lossless=True)
            except:
                continue

if __name__ == "__main__":
    main()