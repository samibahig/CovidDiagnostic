# 🫁 CovidDiagnostic — COVID-19 Detection from Cough Audio

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![SpeechBrain](https://img.shields.io/badge/SpeechBrain-Audio%20AI-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Non-invasive COVID-19 diagnostic from cough audio using **ECAPA-TDNN** — a state-of-the-art speaker/sound embedding architecture.

---

## 🎯 Project Overview

Can COVID-19 be detected from a cough recording?

This project explores **audio-based COVID-19 diagnosis** using deep learning — a non-invasive, low-cost screening approach that could complement traditional testing, especially in resource-limited settings.

```
Cough Audio Recording
        │
        ▼
  Audio Preprocessing
  (MiniLibrispeech pipeline)
        │
        ▼
  ECAPA-TDNN Embeddings
  (End-to-End deep audio model)
        │
        ▼
  COVID-19 Classification
  (Positive / Negative)
```

---

## 🧠 Architecture — ECAPA-TDNN

**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation) is a state-of-the-art architecture for audio classification, originally designed for speaker verification — here applied to pathological sound detection.

Key advantages:
- Captures both local and global temporal patterns in audio
- Attention mechanism focuses on diagnostically relevant cough features
- Strong performance on short audio clips

---

## 🗂️ Project Structure

```
CovidDiagnostic/
│
├── 🐍 preparation/         ← Data preparation pipeline (MiniLibrispeech)
├── ⚙️ ECAPA/               ← ECAPA-TDNN model architecture
├── 📋 train.yaml           ← Hyperparameters configuration
├── 🐍 train.py             ← Main training script
└── 📖 README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/samibahig/CovidDiagnostic.git
cd CovidDiagnostic

# Install dependencies
pip install speechbrain torch torchaudio

# Prepare data
python preparation/prepare_data.py

# Train the model
python train.py train.yaml
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Audio AI Framework | SpeechBrain |
| Deep Learning | PyTorch |
| Architecture | ECAPA-TDNN |
| Audio Processing | torchaudio |

---

## 🔮 Next Steps

- [ ] Benchmark on larger COVID cough datasets (COUGHVID, Coswara)
- [ ] Compare with baseline models (x-vectors, i-vectors)
- [ ] Add explainability — visualize which cough features drive predictions
- [ ] Clinical validation with medical team

---

## 👤 Author

**Sami Bahig, MD MSc** — Data Scientist & AI Engineer
Université de Montréal / MILA

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/samibahig)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/samibahig)

---

## 🔗 Related Projects

| Project | Description | Link |
|---|---|---|
| RecoverProject | Long COVID — Metabolomic & proteomic ML | [GitHub](https://github.com/samibahig/RecoverProject) |
| FAERS 2025 | FDA pharmacovigilance — 28M records | [GitHub](https://github.com/samibahig/faers-2025-pharmacovigilance) |
| Protocol TDM | OCR + CamemBERT — Radiology classification | [GitHub](https://github.com/samibahig/Prediction-Image-Protocole-) |

---

*MIT License · Sami Bahig · 2021*
