# DermAI: Skin Cancer Detection Using CNN and Transfer Learning

This project leverages a Convolutional Neural Network (CNN) with transfer learning to accurately diagnose external skin lesions from a single image. The goal is to distinguish between benign and malignant skin conditions using deep learning, facilitating early detection and intervention. 

The relevant files with results are as follows: 
* `modelruns.ipynb` - contains the primary model
*  `baselineModel.ipynb` - contains functions for preprocessing and the baseline model
* `testing.ipynb` and `testing_updated.ipynb` - contains functions to import HIBA test dataset and evaluate the best model on the new data

Preprocessed images were downloaded to the folder `aps360dataset` after running preprocessing functions in [this](https://colab.research.google.com/drive/1XqBiA8LButjDnEbptTXUhJLl129Eed09?usp=sharing) Colab notebook

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Data Sources](#data-sources)
- [Model Architecture](#model-architecture)
- [Pre-processing Pipeline](#pre-processing-pipeline)
- [Results](#results)
- [Installation](#installation)

## Introduction
Skin cancers are one of the most common forms of cancer globally, with early detection playing a critical role in improving patient outcomes. This project uses a CNN-based approach to classify external skin lesions into four classes: melanoma, nevus, basal cell carcinoma (bcc), and benign keratosis-like lesions (bkl).

## Motivation
Early diagnosis of skin cancers can drastically increase survival rates, but diagnosis is often delayed due to limited access to dermatologists and biases in manual examination, particularly across different skin tones. This project aims to reduce diagnostic delays by providing an accessible, automated classification tool that works across all skin types.

## Data Sources
The model is trained on two primary datasets:
1. **HAM10000**: A collection of 10,015 dermoscopic images from Australia and Austria.
2. **ISIC 2019**: A dataset of 25,331 images sourced from Europe, Australia, and North America.

Additionally, an independent test set from the **Hospital Italiano de Buenos Aires** is used to evaluate model performance on a diverse patient population.

## Model Architecture
The core of the project is built on transfer learning with **ResNet-18** for feature extraction, integrated with a **Convolutional Block Attention Module (CBAM)** to enhance feature representation. The model includes:
- **ResNet-18** for initial feature extraction.
- A custom CNN classifier with 3 convolutional layers.
- **CBAM** for channel and spatial attention.
- **Dropout** and **Batch Normalization** for regularization.

For full details on the architecture, refer to the [report](DermAI-report.pdf).

## Pre-processing Pipeline
The pre-processing pipeline ensures data quality and consistency:
- **Hair and noise removal** using the "dull razor" technique and Gaussian filtering.
- **Image resizing** to 224x224 pixels for ResNet input.
- **Data augmentation** (flipping and rotation) to address class imbalance.

For more details on the pipeline, refer to the documentation [here](DermAI-report.pdf).

## Results
The model achieved:
- **73.7% accuracy** on the HAM10000/ISIC test set.
- **68.9% accuracy** on the unseen Buenos Aires dataset.
- Precision and recall scores of over 0.7 for cancerous lesions, minimizing false negatives, which is critical in early skin cancer detection.

For more details on the results and performance evaluation, check out the [report](DermAI-report.pdf).

## Installation
Clone the repository:
```bash
git clone https://github.com/reaahuja/DermAI-Skin-Cancer-Detection.git
cd DermAI-Skin-Cancer-Detection

