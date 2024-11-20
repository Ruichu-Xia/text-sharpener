
### Checkpoint Structure
```
{
    "epoch": epoch,                     
    "model_state_dict": model.state_dict(),
    "history": history                 
}
```

### Load checkpoint

```
NUM_CHANNELS = 1

model = SharpNet(channels=NUM_CHANNELS)
model = model.to(device)
```

```
ckpt_path = "./ckpt/ckpt_700"
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint["model_state_dict"])
```

### Additional Info
```
epoch = checkpoint["epoch"]          # saved epoch
history = checkpoint["history"]      # training history (loss, PSNR)
```
