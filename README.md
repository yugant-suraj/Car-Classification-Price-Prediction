# 🎥 Smart Surveillance System (OpenFace)

> Real-time face recognition system using OpenFace facial embeddings and SVM-based classification.

---

## Overview
This project implements a **real-time smart surveillance system** capable of detecting and recognizing faces from a live video stream. It uses **OpenFace (FaceNet-based embeddings)** to extract discriminative facial features and a **Support Vector Machine (SVM)** classifier for identity recognition.

The system is designed with a **modular pipeline**, enabling easy model replacement, scalability, and real-world deployment.

---

## System Architecture

<pre>
Video Stream
  |
  v
Face Detection (OpenCV DNN)
  |
  v
Embedding Extraction (OpenFace)
  |
  v
128-D Feature Vector
  |
  v
SVM Classifier
  |
  v
Identity + Confidence Score
</pre>




---

## Key Features
- Real-time face detection from live camera feed
- Deep facial embedding extraction using OpenFace
- SVM-based identity classification
- Confidence score for each prediction
- Modular pipeline (embedding, training, recognition separated)

---

## Tech Stack
- Python 3.6+
- OpenCV (DNN module)
- PyTorch (OpenFace model)
- scikit-learn (SVM classifier)
- NumPy
- imutils
- Pickle (model persistence)

---

## Repository Structure

<pre>
.
├── embeddings_extraction.py
├── train.py
├── identification.py
├── embeddings.pickle
├── recognizer.pickle
├── le.pickle
├── Major_Project_(V_2_5).ipynb
└── README.md
</pre>

---

## Setup
Install required dependencies:

```bash
pip install opencv-python torch scikit-learn imutils numpy
```

## Reference

FaceNet: A Unified Embedding for Face Recognition and Clustering (CVPR 2015)

