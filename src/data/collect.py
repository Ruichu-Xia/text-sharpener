import fitz 
from PIL import Image
import io
import random
import os
import easyocr
import numpy as np

from src.data.utils import CLEAR_DIR, TEXTBOOK_PATH, CROP_SIZE, NUM_CROPS_PER_PAGE


def main(): 
    os.makedirs(CLEAR_DIR, exist_ok=True)
    ocr_reader = easyocr.Reader(['en'])
    pdf_reader = fitz.open(TEXTBOOK_PATH)

    img_idx = 0
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))

        if img.size[0] >= CROP_SIZE and img.size[1] >= CROP_SIZE:
            for _ in range(NUM_CROPS_PER_PAGE):
                x_max = img.size[0] - CROP_SIZE
                y_max = img.size[1] - CROP_SIZE
                x = random.randint(0, x_max)
                y = random.randint(0, y_max)

                cropped_img = img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
                cropped_img = cropped_img.convert('L')

                cropped_img_np = np.array(cropped_img)
                detected_text = ocr_reader.readtext(cropped_img_np)
                if len(detected_text) > 10:  
                    cropped_img.save(f'{CLEAR_DIR}/Image_{img_idx}.png')
                    img_idx += 1

    pdf_reader.close()

if __name__ == "__main__":
    main()