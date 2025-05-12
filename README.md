# Political Deepfake Detection: A Multimodal Proof-of-Concept

**Author:** Jonathan Monroe  
**Course:** MACSS 30200 (Proposal Track)  
**Date:** Spring 2025  

---

## Abstract

This project builds a reproducible pipeline to detect political deepfakes on YouTube by fusing **visual** (face-cropped frames) and **audio** (2-second speech clips) features. We leverage pretrained deep learning models for both modalities and analyze their combined performance and potential political bias.

### Key Components:

1. **Data Collection**  
   - Search for "<candidate> deepfake" via YouTube API  
   - Download real and fake videos using `yt-dlp` with browser cookies

2. **Feature Extraction**  
   - Sample one frame every 2 seconds using OpenCV  
   - Detect and crop faces with 50% margin via `face_recognition`  
   - Extract 2-second audio snippets synchronized with the sampled frames

3. **Modeling**  
   - Visual features: Pretrained Xception (via `timm`)  
   - Audio features: Pretrained Wav2Vec2 (via `transformers`)  
   - Fusion: Logistic regression on concatenated image/audio embeddings

4. **Evaluation**  
   - ROC AUC: Vision-only (0.461) vs. Multimodal (0.990)  
   - ΔFPR (Biden–Trump): ~0.056 false-positive gap  
   - Grad-CAM used to identify spatial attention patterns  
   - Group-based bias metrics and t-tests on attention maps

---

## Features

- End-to-end Jupyter notebook for the full pipeline  
- Automated caching of YouTube API search results  
- Face detection via `face_recognition` + OpenCV  
- Audio clipping with `ffmpeg`  
- Multimodal fusion with scikit-learn  
- Detailed bias metrics and visualization suite  

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── pilot.ipynb                   # Single notebook with full pipeline (data to evaluation)
├── data/
│   ├── urls/                     # Saved YouTube video URLs per candidate
│   └── videos/                   # Downloaded real/fake videos
├── data3/
│   ├── frames_cropped/           # Face-cropped frames organized by candidate/label
│   └── audio_clips/              # Corresponding 2-second audio clips
├── results/
│   ├── vision_scores.csv
│   ├── multimodal_scores.csv
│   └── plots/                    # Visualizations and charts for presentation
└── slides/
    └── presentation.pdf          # Final Beamer presentation slides
```

---

## Installation

Requires **Python 3.10+** and **FFmpeg**.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Requirements

```text
google-api-python-client==2.125.0
pandas==2.2.3
numpy==2.1.3
opencv-python==4.11.0.86
face-recognition==1.3.0
dlib==19.24.6
torch==2.7.0
torchvision==0.22.0
pillow==11.1.0
scikit-learn==1.6.1
transformers==4.40.1
timm==0.9.16
soundfile==0.12.1
captum==0.6.0
shap==0.47.2
graphviz==0.20.3
yt-dlp==2025.4.30
moviepy==1.0.3
matplotlib==3.8.4
```

---

## Usage

All steps—from data collection to modeling and evaluation—are implemented in:

```text
pilot.ipynb
```

To run:

```bash
jupyter notebook pilot.ipynb
```

This will:

1. Query YouTube and download candidate-specific real/fake videos  
2. Sample and crop frames using face detection  
3. Extract synchronized audio using FFmpeg  
4. Generate multimodal embeddings and train logistic regression  
5. Evaluate ROC AUC and compute candidate-specific FPRs  
6. Visualize Grad-CAM heatmaps and decision boundaries  

---

## Results Summary

| Model         | ROC AUC | ΔFPR (Biden–Trump) |
|---------------|---------|--------------------|
| Vision-Only   | 0.461   | 0.031              |
| Multimodal    | 0.990   | 0.056              |

- **Multimodal fusion** drastically improved classification accuracy  
- **ΔFPR** indicates slight political skew in false-positive errors  
- **Grad-CAM COM** analysis showed candidate-specific visual focus areas  

---

## Citation

If you use or build upon this work, please cite:

> Monroe, J. (2025). *Political Deepfake Detection: A Multimodal Proof-of-Concept*. MACSS 30200 Proposal Track.

---

## Contact

For questions, collaboration, or follow-up:

- Email: jonathanmonroe@uchicago.edu  
- GitHub: [github.com/JonathanPMonroe/MACSS-30200](https://github.com/JonathanPMonroe/MACSS-30200)
