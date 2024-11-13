import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


from src.model.unet import UNet
from src.model.cbam_unet import UNetWithCBAM
from src.data_pipeline.dataset import LocalImageDataset, get_split_indices
from src.model.utils import INPUT_DIR, TARGET_DIR, NUM_CHANNELS, FEATURES, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, BATCH_SIZE, TEXT_WEIGHT, WIDTH, DW_EXPAND, FFN_EXPAND, ENC_BLKS, MIDDLE_BLK_NUM, DEC_BLKS, get_device
from src.model.custom_loss import PSNRLoss

     
def train_model(model, train_loader, optimizer, reconstruction_loss, scaler, device): 
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for input_images, target_images in loop: 
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        with torch.autocast(device_type=str(device), dtype=torch.float16):
            output_images = model(input_images)
            loss = reconstruction_loss(output_images, target_images) 

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())  

        total_loss += loss.item()
    avg_train_loss = total_loss / len(loop)
    return avg_train_loss

def validate_model(model, val_loader, reconstruction_loss, device): 
    model.eval()  
    total_val_loss = 0

    with torch.no_grad():  
        for input_images, target_images in val_loader:
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            output_images = model(input_images)
            loss = reconstruction_loss(output_images, target_images) 
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_val_loss_{val_loss:.4f}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")


def main(): 
    device = get_device()
    # model = UNet(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS, features=FEATURES).to(device)
    model = UNetWithCBAM(img_channel=NUM_CHANNELS, 
                         width=WIDTH, 
                         middle_blk_num=MIDDLE_BLK_NUM, 
                         enc_blk_nums=ENC_BLKS, 
                         dec_blk_nums=DEC_BLKS, 
                         dw_expand=DW_EXPAND, 
                         ffn_expand=FFN_EXPAND,)
    reconstruction_loss = PSNRLoss()

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    num_images = len([f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))])
    num_images = 32
    train_indices, val_indices, test_indices = get_split_indices(num_images)

    print(train_indices)
    train_loader = DataLoader(LocalImageDataset(INPUT_DIR, TARGET_DIR, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LocalImageDataset(INPUT_DIR, TARGET_DIR, val_indices), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(LocalImageDataset(INPUT_DIR, TARGET_DIR, test_indices), batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float('inf')
    checkpoint_dir = "checkpoints"

    for epoch in range(1, NUM_EPOCHS + 1): 
        avg_train_loss = train_model(model, train_loader, optimizer, reconstruction_loss, scaler, device)
        avg_val_loss = validate_model(model, val_loader, reconstruction_loss, device)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
        
        scheduler.step(avg_val_loss)

if __name__ == "__main__":
    main()
