import random
import numpy as np
import cv2
from PIL import Image
import os

from src.data.utils import CLEAR_DIR, BLURRED_DIR


def apply_motion_blur(image: Image.Image) -> Image.Image:
    kernel_size = random.choice([3, 6, 9])
    angles = list(range(0, 181, 15))
    angle = random.choice(angles)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0), (kernel_size, kernel_size))
    kernel /= kernel_size

    image_np = np.array(image)
    blurred = cv2.filter2D(image_np, -1, kernel)
    return Image.fromarray(blurred)

def main():
    os.makedirs(BLURRED_DIR, exist_ok=True)
    for image_name in os.listdir(CLEAR_DIR):
        if image_name.lower().endswith('.png'): 
            image_path = os.path.join(CLEAR_DIR, image_name)
            image = Image.open(image_path)
            

            blurred_image = apply_motion_blur(image)
            output_path = os.path.join(BLURRED_DIR, image_name)
            blurred_image.save(output_path, format="png", lossless=True)



if __name__ == "__main__":
    main()