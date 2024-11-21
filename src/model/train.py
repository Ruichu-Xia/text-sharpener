import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math


### Edge loss
# -----
class SobelFilter(torch.nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Sobel kernels as buffers
        self.register_buffer("sobel_x", torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
        ).unsqueeze(1))  # Shape: [1, 1, 3, 3]

        self.register_buffer("sobel_y", torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
        ).unsqueeze(1))  # Shape: [1, 1, 3, 3]

    def forward(self, img):
        sobel_x = self.sobel_x.to(dtype=img.dtype, device=img.device)
        sobel_y = self.sobel_y.to(dtype=img.dtype, device=img.device)
        
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # epsilon for stability

sobel_filter = SobelFilter()

def edge_loss(pred, target):
    pred_edges = sobel_filter(pred)
    target_edges = sobel_filter(target)
    return F.mse_loss(pred_edges, target_edges)

def combined_loss(pred, target, reconstruction_loss_fn, edge_weight=0.1):
    reconstruction_loss = reconstruction_loss_fn(pred, target)
    edge_loss_value = edge_loss(pred, target)
    return reconstruction_loss + (edge_weight * edge_loss_value)
### -----

def psnr(output, target):
    mse = F.mse_loss(output, target)
    psnr_value = 20 * math.log10(1.0) - 10 * math.log10(mse.item())  # assume inputs normalized to [0, 1]
    return psnr_value

def train_model(model, train_loader, optimizer, reconstruction_loss, scaler, grad_clip=None, edge_weight=0, device=None): 
    model.train()
    total_loss = 0
    total_psnr = 0
    num_batches = len(train_loader)

    for batch_idx, (input_images, target_images) in enumerate(train_loader): 
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        with torch.autocast(device_type=str(device), dtype=torch.float16):
            output_images = model(input_images)
            loss = combined_loss(output_images, target_images, reconstruction_loss, edge_weight)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        ### gradient clipping
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        ### ----------

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_psnr += psnr(output_images, target_images)

    avg_train_loss = total_loss / num_batches
    avg_train_psnr = total_psnr / num_batches

    return avg_train_loss, avg_train_psnr


def validate_model(model, val_loader, reconstruction_loss, device, edge_weight=0.1): 
    model.eval()  
    total_val_loss = 0
    total_psnr = 0

    with torch.no_grad():  
        for input_images, target_images in val_loader:
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            output_images = model(input_images)
            loss = combined_loss(output_images, target_images, reconstruction_loss, edge_weight)
            total_val_loss += loss.item()
            total_psnr += psnr(output_images, target_images)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    return avg_val_loss, avg_psnr


### -----
def save_checkpoint(epoch, model, history, checkpoint_dir, test_indices=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_indices": test_indices,
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")


def save_samples(epoch, model, val_loader, device, output_dir, sample_loader=None, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    samples = []

    # use fixed sample loader if provided
    loader = sample_loader if sample_loader is not None else val_loader

    with torch.no_grad():
        for _, (input_images, target_images) in enumerate(loader):
            input_images = input_images.to(device)
            denoised_images = model(input_images)

            # input, denoised, and target images
            for i in range(min(len(input_images), num_samples - len(samples))):
                samples.append((input_images[i].cpu(), denoised_images[i].cpu(), target_images[i].cpu()))

            if len(samples) >= num_samples:
                break

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i, (input_img, denoised_img, target_img) in enumerate(samples):
        axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 0].set_title("Input (Blurry)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(denoised_img.permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 1].set_title("Model Denoised")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(target_img.permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 2].set_title("Target (Ground Truth)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"epoch_{epoch}_samples.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {num_samples} samples at epoch {epoch} to {save_path}")
