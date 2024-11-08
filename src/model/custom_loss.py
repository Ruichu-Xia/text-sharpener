from difflib import SequenceMatcher

from src.model.utils import rescale_and_convert_image, get_device
import easyocr 

device = get_device()
reader = easyocr.Reader(["en"], gpu=False if device == "cpu" else True)

def text_accuracy_loss(output_images, target_images): 
    output_image_numpy = rescale_and_convert_image(output_images)
    target_image_numpy = rescale_and_convert_image(target_images)
    
    batch_size = len(output_image_numpy)
    loss = 0

    for i in range(batch_size): 
        output_current = output_image_numpy[i]
        target_current = target_image_numpy[i]

        print(f"Type of output_current: {type(output_current)}")
        print(f"Shape of output_current: {output_current.shape if hasattr(output_current, 'shape') else 'No shape attribute'}")

        print(f"Type of target_current: {type(target_current)}")
        print(f"Shape of target_current: {target_current.shape if hasattr(target_current, 'shape') else 'No shape attribute'}")

        output_text = " ".join(reader.readtext(output_current, detail=0))
        target_text = " ".join(reader.readtext(target_current, detail=0))

        loss += 1 - SequenceMatcher(None, output_text, target_text).ratio()

    return loss / batch_size



    





