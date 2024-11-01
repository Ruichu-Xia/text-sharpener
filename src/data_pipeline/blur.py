import random
import numpy as np
import cv2
from PIL import Image
import os
from utils import PROCESSED_IMAGE_DIR, BLURRED_IMAGE_DIR

# gaussian blur, motion blur, speckle noise, jpeg compression artifact
def apply_gaussian_blur(image: Image.Image) -> Image.Image:
    kernel_size = random.choice([9, 15, 21, 27, 33])
    sigma = random.choice([1.5, 2.5, 3.5, 4.5, 5.5])
    image_np = np.array(image)
    blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
    return Image.fromarray(blurred)

def apply_motion_blur(image: Image.Image) -> Image.Image:
    kernel_size = random.choice([9, 15, 21, 27, 33])
    angles = list(range(0, 181, 15))
    angle = random.choice(angles)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0), (kernel_size, kernel_size))
    kernel /= kernel_size

    image_np = np.array(image)
    blurred = cv2.filter2D(image_np, -1, kernel)
    return Image.fromarray(blurred)

def apply_compression_artifacts(image: Image.Image) -> Image.Image:
    quality = random.randint(1, 25)
    temp_path = "temp_low_quality.jpg"
    image.save(temp_path, format="JPEG", quality=quality)
    compressed_image = Image.open(temp_path)
    os.remove(temp_path)
    return compressed_image

def apply_speckle_noise(image: Image.Image) -> Image.Image:
    intensity = random.uniform(0.05, 0.1)
    image_np = np.array(image) / 255.0
    noise = np.random.normal(0, intensity, image_np.shape)
    noisy_image = np.clip(image_np + noise, 0, 1) * 255
    return Image.fromarray(noisy_image.astype(np.uint8))


def apply_random_blur(image: Image.Image) -> Image.Image:
    blurred_image =  image.copy()

    if random.choice([True, False]):
        blurred_image = apply_gaussian_blur(blurred_image)
    else: 
        blurred_image = apply_motion_blur(blurred_image)
    
    if random.choice([True, False]):
        blurred_image = apply_compression_artifacts(blurred_image)
    
    if random.choice([True, False]):
        blurred_image = apply_speckle_noise(blurred_image)
    
    return blurred_image


def main():
    for image_name in os.listdir(PROCESSED_IMAGE_DIR):
        image_path = os.path.join(PROCESSED_IMAGE_DIR, image_name)
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        blurred_image = apply_random_blur(image)
        output_path = os.path.join(BLURRED_IMAGE_DIR, image_name)
        blurred_image.save(output_path, format="WEBP", lossless=True)


if __name__ == "__main__":
    main()