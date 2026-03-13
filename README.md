# 🫁 CovidDiagnostic — COVID-19 Detection from Cough Audio

> **Non-invasive COVID-19 screening from cough recordings using ECAPA-TDNN deep learning architecture.**  
> A low-cost, privacy-preserving diagnostic tool designed to complement traditional testing — especially in resource-limited settings.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![SpeechBrain](https://img.shields.io/badge/SpeechBrain-FF6B35?style=flat-square)
![License](https://img.shields.io/badge/License-Domain-lightgrey?style=flat-square)

---

## 🎯 Project Overview

**Can COVID-19 be detected from a cough recording?**

This project explores audio-based COVID-19 diagnosis using deep learning. The goal is to develop a non-invasive, scalable screening approach that can:

- Complement PCR and antigen testing as a rapid first-pass screen
- Be deployed remotely via smartphone or web interface
- Operate without exposing sensitive patient data
- Scale to resource-limited or remote clinical environments

---

## 🔬 Clinical Context

| Consideration | Detail |
|---|---|
| **Screening Gap** | Demand for rapid, accessible COVID testing outpaced lab capacity during peak pandemic phases |
| **Audio Biomarkers** | COVID-19 affects the respiratory tract, producing characteristic cough patterns detectable by AI |
| **Non-invasiveness** | Audio recording requires no physical contact, reagents, or medical personnel |
| **Privacy-preserving** | No biometric identity data — only acoustic features are extracted |
| **Complementarity** | Designed as a pre-screening tool, not a replacement for confirmed clinical testing |

---

## 📊 Pipeline Overview

```
Cough Audio Recording
        │
        ▼
  Audio Preprocessing
  (MiniLibrispeech pipeline)
  ─ Resampling, normalization, segmentation
        │
        ▼
  Feature Extraction
  ─ Log-Mel filterbanks
  ─ MFCC-style representations
        │
        ▼
  ECAPA-TDNN Embeddings
  ─ Channel attention & propagation
  ─ Multi-scale aggregation
  ─ Attentive statistical pooling
        │
        ▼
  COVID-19 Classification
  (Positive / Negative)
  ─ Softmax output
  ─ Confidence score
```

---

## 🧠 Architecture — ECAPA-TDNN

**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation Time-Delay Neural Network) is a state-of-the-art architecture for audio embedding, originally designed for speaker verification — here repurposed for pathological sound detection.

### Key Design Advantages

| Feature | Benefit for COVID Detection |
|---|---|
| **Channel Attention** | Emphasizes the most informative frequency bands of cough sounds |
| **Multi-scale Propagation** | Captures both short bursts and longer cough patterns |
| **Attentive Pooling** | Focuses on diagnostically relevant temporal segments |
| **Compact Embeddings** | Efficient inference on short audio clips (< 5 seconds) |
| **Pre-training Transfer** | Leverages representations from large-scale audio corpora |

### Why ECAPA-TDNN over alternatives?

- **vs. x-vectors**: More expressive with attention-based pooling; better on short clips
- **vs. i-vectors**: End-to-end trainable; no GMM-UBM required
- **vs. CNNs**: Better temporal modeling via TDNN layers
- **vs. Transformers**: Lower compute requirements; better suited for audio classification at this scale

---

## 🗂️ Project Structure

```
CovidDiagnostic/
│
├── 🐍 preparation/         ← Data preparation pipeline
│   └── prepare_data.py     ← MiniLibrispeech-style preprocessing
│
├── ⚙️ ECAPA/               ← ECAPA-TDNN model architecture
│   └── model.py            ← Full model definition (SpeechBrain)
│
├── 📋 train.yaml           ← Hyperparameters & experiment config
├── 🐍 train.py             ← Main training script
└── 📖 README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/samibahig/CovidDiagnostic.git
cd CovidDiagnostic
```

### 2. Install dependencies

```bash
pip install speechbrain torch torchaudio
```

### 3. Prepare data

```bash
python preparation/prepare_data.py
```

> Ensure your cough audio dataset is available and paths are configured in `train.yaml`.

### 4. Train the model

```bash
python train.py train.yaml
```

### 5. Evaluate

```bash
# Example inference (adapt to your evaluation script)
python train.py train.yaml --eval
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Audio AI Framework** | SpeechBrain |
| **Deep Learning** | PyTorch |
| **Architecture** | ECAPA-TDNN |
| **Audio Processing** | torchaudio |
| Data Pipeline | MiniLibrispeech-style JSON manifests |

---

## 🔮 Next Steps

- [ ] **Benchmark on larger datasets** — COUGHVID, Coswara, Cambridge COVID Sound
- [ ] **Baseline comparison** — x-vectors, i-vectors, standard TDNN
- [ ] **Explainability** — Grad-CAM on spectrograms; attention map visualization
- [ ] **Data augmentation** — SpecAugment, noise injection, room impulse response
- [ ] **Multi-class extension** — COVID vs. influenza vs. healthy cough
- [ ] **Clinical validation** — Sensitivity / specificity evaluation with medical team
- [ ] **Deployment** — FastAPI inference endpoint + lightweight web interface

---

## 📚 References

- Desplanques et al. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech 2020.
- Raavan et al. (2021). *Coswara — A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis.* Interspeech 2021.
- Orlandic et al. (2021). *The COUGHVID crowdsourcing dataset.* Scientific Data.
- Radhakrishnan et al. (2021). *Exploring Automatic Diagnosis of COVID-19 from Crowdsourced Respiratory Sound Data.* KDD 2021.



## 👤 Author

**Sami Bahig, MD MSc** — Data Scientist & AI Engineer  
Université de Montréal / MILA

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/samibahig)
[![GitHub](https://img.shields.io/badge/GitHub-samibahig-181717?style=flat-square&logo=github)](https://github.com/samibahig)

---

*MIT License · Sami Bahig · 2021*

