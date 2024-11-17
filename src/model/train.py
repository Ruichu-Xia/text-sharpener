import torch
from tqdm import tqdm
import os


     
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
    checkpoint_path = os.path.join(checkpoint_dir, f"test_model_epoch_{epoch}_val_loss_{val_loss:.4f}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

    
