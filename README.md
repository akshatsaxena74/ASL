# ASL Fingerspelling Recognition

This repository contains a project to recognise the fingerspelled letters of American Sign Language (ASL) from images or video frames. By training a machine learning model on labelled images of each letter, users can predict the corresponding English alphabet character.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Datasets](#datasets)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
---

## Introduction

ASL is a visual language used widely in the Deaf community. Fingerspelling uses hand shapes to represent letters of the English alphabet. This project demonstrates how to train a convolutional neural network (CNN) or use transfer learning to classify these ASL signs accurately.

---

## Features

- **Image Classification**: Recognize static ASL letters from images.
- **Real-Time Prediction** (optional): Integrate with a webcam or video feed.
- **Data Augmentation**: Improve generalization through random transformations.
- **Customizable Model**: Swap or fine-tune architecture as needed.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/ASL-Fingerspelling.git
   cd ASL-Fingerspelling
   ```

2. **Create and Activate a Virtual Environment** (recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS/Linux
   venv\Scripts\activate        # Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Jupyter Notebook** (optional)  
   ```bash
   pip install jupyter
   jupyter notebook
   ```

---

## Usage

1. **Training**  
   - Open and run the provided notebook (e.g., `ASL_Fingerspelling.ipynb`).
   - Modify dataset paths as needed.
   - Adjust hyperparameters and run all cells to train.

2. **Testing**  
   - Evaluate on a held-out test set or your custom dataset.
   - Optionally, use the webcam script (if provided) for real-time sign detection.

3. **Inference**  
   - After training, you can run:
     ```bash
     python inference.py --image <path_to_image>
     ```
     or
     ```bash
     python inference.py --video
     ```
     For real-time prediction.

---

## Datasets

Common sources include:
- [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet)
- Public or user-collected images of ASL letters.

---

## Model Architecture

- **Custom CNN** or **Pretrained Model** (e.g., ResNet, MobileNet).
- **Layers**: Convolutional, Pooling, Fully-Connected layers for classification.
- **Optimizer**: Commonly `Adam` or `SGD`.
- **Loss Function**: Categorical cross-entropy for multi-class classification.

---

## Results

- **Training Accuracy**: 97%
- **Validation Accuracy**: 95%
- **Test Accuracy**: 93%
  
---

## Contributors 
Akshay Reddy, Akshat Srivastava, Akshat Saxena & Akshat Swarup
