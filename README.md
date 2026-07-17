# Bone-Fracture-Detection-using-Deep-Learning-Models

## 📋 Project Overview
This repository contains a deep learning-based pipeline designed to automate the detection of bone fractures from radiographic (X-ray) images. By leveraging computer vision, this project aims to assist in preliminary diagnostic triage, accurately classifying images as **fractured** or **non-fractured**.

## 📊 Dataset Specifications
The model is trained on the [Bone Fracture Multi-Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/data), which comprises 10,580 high-quality radiographic images covering diverse anatomical regions, including the lower limb, upper limb, lumbar, hips, and knees.

| Split | Number of Images |
| :--- | :--- |
| **Training** | 9,246 |
| **Validation** | 828 |
| **Test** | 506 |

## 🚀 Technical Implementation
*   **Pipeline:** The workflow follows an end-to-end computer vision approach, incorporating advanced image preprocessing (resizing, normalization, and augmentation) to ensure model robustness.
*   **Architecture:** The project employs deep learning models optimized for binary classification to identify fracture-related anomalies effectively.
*   **Evaluation:** Performance is rigorously validated using standard metrics, including accuracy, precision, and recall, to ensure diagnostic reliability.

## 🤝 Acknowledgments
This dataset is a compilation of research contributions from:
*   [Mohan Kumar](https://www.kaggle.com/datasets/amohankumar/bone-break-classifier-dataset) (Bone Break Classifier Dataset)
*   [Abdelaziz Faramawy](https://www.kaggle.com/datasets/abdelazizfaramawy/bone-fracture) (bone_fracture)
*   [Harsha Arya](https://www.kaggle.com/datasets/harshaarya/fracture) (fracture)

## ⚖️ License
This project is licensed under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/pddl/1.0/). Users are permitted to use, share, and modify this work, provided that appropriate credit is given to the original creators and the dataset source.

---

### Getting Started
To reproduce the results, clone this repository and install the necessary dependencies:

```bash
git clone [https://github.com/Sai-Mohith24/Bone-Facture-detection-using-Deep-Learning-Models.git](https://github.com/Sai-Mohith24/Bone-Facture-detection-using-Deep-Learning-Models.git)
pip install -r requirements.txt
python train.py
