import torch
from difflib import SequenceMatcher

from src.model.utils import rescale_and_convert_image, get_device
import easyocr 

device = get_device()
reader = easyocr.Reader(["en"], gpu=False if device == "cpu" else True)

def text_accuracy_loss(output_images, target_images): 
    output_images = rescale_and_convert_image(output_images)
    target_images = rescale_and_convert_image(target_images)
    
    batch_size = len(output_images)
    loss = 0

    for i in range(batch_size): 
        output_current = output_images[i]
        target_current = target_images[i]

        output_text = " ".join(reader.readtext(output_current, detail=0))
        target_text = " ".join(reader.readtext(target_current, detail=0))

        loss += 1 - SequenceMatcher(None, output_text, target_text).ratio()

    return torch.tensor(loss / batch_size, dtype=torch.float32, device=device)



    





