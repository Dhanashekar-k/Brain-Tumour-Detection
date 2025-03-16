# Brain Tumor Detection 

## Overview
This project is dedicated to brain tumor detection and segmentation using magnetic resonance imaging (MRI) data from the BRATS 2021 challenge. It aims to develop and evaluate advanced deep learning models capable of accurate brain tumor segmentation and classification.

## Dataset Description
The BRATS (Brain Tumor Segmentation) 2021 dataset provides multi-modal MRI scans (T1, T1ce, T2, and FLAIR) annotated for brain tumor segmentation tasks. The dataset is widely used as a benchmark for developing computational models in medical imaging.
•	Data Types: Multi-modal MRI scans (T1, T1ce, T2, FLAIR)
•	Annotations: Manual expert annotations indicating tumor regions (necrotic core, edema, enhancing tumor)
•	Task: Segmentation and classification of tumor tissues into distinct categories

## Models Evaluated
### 3D UNet
3D variant of the traditional UNet architecture adapted for volumetric data.
### Nested 3D UNet
Enhanced UNet architecture with dense skip connections to improve segmentation performance.
### ConvGLU Nested UNet with Attention
Nested UNet structure incorporating convolutional gated linear units (ConvGLU) and attention mechanisms to enhance segmentation performance.

Detailed training logs for each model can be accessed here: [link-to-logs-folder].

## Project Structure
project_root/
├── models/
│   ├── dataloader.py
│   └── training.py
├── 3d_unet.py
├── nested_3d_unet.py
├── convglu_nested_unet_attention.py
└── logs/
    ├── 3d_unet_logs
    ├── nested_3d_unet_logs
    └── convglu_nested_unet_attention_logs
    
## Usage
Run the individual Python scripts (3d_unet.py, nested_3d_unet.py, convglu_nested_unet_attention.py) to train and evaluate respective models. Training configurations and hyperparameters are managed in the corresponding training scripts and logged in the respective logs directory.
