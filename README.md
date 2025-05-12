# Political Deepfake Detection: A Multimodal Proof-of-Concept

**Author:** Jonathan Monroe  
**Course:** MACSS 30200 (Proposal Track)  
**Date:** Spring 2025  

---

## Abstract

We build a reproducible pipeline to detect political deepfakes on YouTube by fusing **vision** (face-cropped frames) and **audio** (2 s speech clips) features.  
1. **Data Collection:** “<candidate> deepfake” YouTube searches → download real & fake videos.  
2. **Feature Extraction:** Sample face crops at 0.5 FPS + extract synchronized 2 s audio.  
3. **Modeling:** Pretrained Xception (vision) + Wav2Vec2 (audio) → logistic-regression fusion.  
4. **Bias Analysis:** Audio branch flags real Biden clips ≫ real Trump clips (ΔFPR≈5.6%), revealing political bias.

---

## Repository Structure

```
.
├── README.md                        ← this file
├── requirements.txt                 ← exact package versions
├── notebook: pilot.ipynb
│   ├── 01_data_collection    ← URL search & video download
│   ├── eature_extraction  ← frame & audio extraction
│   ├── model_training  ← vision/audio/fusion training
│   ├── bias_analysis    ← ΔFPR, Grad-CAM, t-tests
│   └── visualizations     ← all final plots
├── data3/                            
│   ├── urls/{fake real}/
│   ├── videos/{candidate}/{real,fake}/             
│   ├── frames_cropped/{Biden,Trump}/{Rframes,Fframes}/
│   └── audio_clips/{Biden,Trump}/{real,fake}/
├── csv
│   ├── multimodal scores
│   └── vision scores
├── Presentation Slides

```

---

## 💻 Requirements

- **Python:** 3.10+  
- **Install:**  
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

```
face_recognition==1.2.3
google-api-python-client
moviepy==2.1.2
numpy==2.1.1
opencv-python==4.11.0
pandas==2.2.2
scikit-learn==1.6.1
timm
torch>=2.0
transformers
soundfile
captum
shap
yt-dlp
```

---

## 🚀 Reproducibility

All steps run top-to-bottom in the notebooks:

1. **Data Collection**  
   Run `notebooks/01_data_collection.ipynb` to build `data/urls/` and download videos into `data/videos/`.

2. **Feature Extraction**  
   Run `notebooks/02_feature_extraction.ipynb` to generate face crops in `data3/frames_cropped/` and audio clips in `data3/audio_clips/`.

3. **Model Training**  
   Run `notebooks/03_model_training.ipynb` to train vision-only, audio-only, and fusion models. Outputs → `results/vision_scores.csv` & `results/multimodal_scores.csv`.

4. **Bias Analysis**  
   Run `notebooks/04_bias_analysis.ipynb` to compute group-wise ROC AUC, ΔFPR, Grad-CAM center-of-mass, and t-tests.

5. **Visualizations**  
   Run `notebooks/05_visualizations.ipynb` to regenerate all figures in `results/plots/`.

---

## 📑 Citation

If you use or extend this work, please cite:

> Monroe, J. (2025). _Political Deepfake Detection: A Multimodal Proof-of-Concept_. MACSS 30200 Proposal Track.

---

## 🔗 GitHub Repository

https://github.com/YOUR_USERNAME/political-deepfake-detection  
*(Replace with your actual repo URL)*
