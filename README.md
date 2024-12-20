<a name="readme-top"></a>

# Text Sharpener

This repository contains the implementation of a Convolutional Neural Network (CNN) model to denoise and deblur text images, improving their readability and clarity. The model extends the U-Net architecture and is trained on synthetic blurry images, optimizing a combined objective of reconstruction and edge loss. 

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
    <li>
      <a href="#dataset">Dataset</a>
    <li>
      <a href="#architecture">Architecture</a>
    <li>
      <a href="#results">Results</a>
    <li>
      <a href="#usage">Usage</a>
  <ol>
</details>

<a name="overview"></a>
## Overview

Blurry text in images poses significant challenges for readability and information extraction, often caused by noise, motion blur, or poor lighting conditions. This project aims to address these issues by developing a CNN-based solution that significantly enhances text image quality.

Key features include: 
- Development of a modified U-Net architecture tailored for text deblurring tasks
- Implemented a combined objective of reconstruction and edge loss
- Improved Peak Signal-to-Noise Ratio (PSNR) on the testing set by **62%**, from **16.97 dB** to **27.21 dB**.

<a name="dataset"></a>
## Dataset
The dataset was sourced from the textbook Pattern Recognition and Machine Learning (2006) by Christopher Bishop. Motion blur was randomly applied to generate input and target pairs. 
![Dataset](./src/visualizations/sample_data.png)

<a name="architecture"></a>
## Architecture 
The model builds upon the **U-Net** architecture with modifications to enhance performance:
- Residual connections added to double convolution blocks for better gradient flow.
- Average pooling used instead of max pooling in downsampling layers to retain context.
- Padding applied to convolutional layers to prevent spatial size reduction.
- Global skip connection directly from input to output for residual learning.

![Architecture](./src/visualizations/unetv2_architecture.png)

<a name="results"></a>
## Results
Improved from **16.97 dB** to **27.21 dB** on the test set.
![Results](./src/visualizations/unetv2_result_visualization.png)


<a name="usage"></a>
## Usage
### Data Preparation 
`src/data/collect.py`: Collect text images from PDFs. 

`src/data/blur.py`: Generate synthetic blurry images.
### Models
`UNetV2.py`: Best-performing model located in `src/model`. Other attempted models like `BasicNet`, `BetterNet`, and `SharpNet` are also included.
### Training 
`src/notebooks/train.ipynb`: Notebook to train models. Ran on Google Colab with T4 GPU.
### Evaluation 
`src/notebooks/evaluation.ipynb`: Notebook to evaluate and visualize model performance.
### Pretrained Model
 Pretrained `UNetV2` model available in `src/ckpt/unetv2_1000`. Test it with:
 ```
import torch
from src.model.UNetV2 import UNetV2

ckpt = torch.load("src/ckpt/unetv2_1000", map_location=torch.device('cpu'))
model = UNetV2(in_channels=1, out_channels=1, features=[64, 128])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
