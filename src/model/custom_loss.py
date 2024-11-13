import torch
import torch.nn as nn
import numpy as np
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

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

    





