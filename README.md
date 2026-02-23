# AAAI-2026-Official-Implementation-Removing-Box-Free-Watermarks
This is the official implementation of ''Removing Box-Free Watermarks for Image-to-Image Models via Query-Based Reverse Engineering'', AAAI 2026.
We only opensource the code for image generation task to avoid code redundancy.

## Repository Structure

The codebase is organized into the following directories:

- `victim_wu/`: Code related to the victim_wu.
- `victim_zhang/`: Code related to the victim_zhang.

Key scripts in each victim directory:
- `forward_HNet.py`: Forward HNet training.
- `inversion_HNet.py`: Inverse HNet training.
- `attack_FHNet.py`: Script for watermark removal attack using Forward HNet.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- tensorboardX
- numpy
- Pillow (PIL)

## Model Checkpoints

The pre-trained checkpoints for both the **Victim Models** and the **Proposed Models** (this paper) are available for download.
- **Victim Models**:
   Download link: [Google Drive Folder](https://drive.google.com/drive/folders/1FLBsGzRa6Y2dFtKjKaG8fSfsibNA4Q_y?usp=drive_link)
- **Proposed Models**:
  Download link: [Google Drive Folder](https://drive.google.com/drive/folders/149A05GwgUYQhohRh4D8REZ62EV0z3Qwf?usp=drive_link)

Please download the models and place them in the appropriate directories.

## Citation
