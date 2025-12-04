<h1 align="center">Bhasha Detector</h1>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/Arka-Dutta-28/bhasha-detector?style=for-the-badge" alt="last commit">
  <img src="https://img.shields.io/badge/languages-22-informational?style=for-the-badge" alt="languages">
</p>

<p align="center">
  <i>Built with the tools and technologies:</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-E14E1C?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white" alt="Supabase">
</p>

---

### *A lightweight transformer-based system for identifying 22 Indic languages (native + romanized)*

🔗 **Live Demo:** https://indic-language-identification.streamlit.app/  
📦 **Dataset Used:** Bhasha-Abhijnaanam  
🧠 **Model Size:** ~40 MB  
📊 **Accuracy:** 87%  
📈 **Macro-F1:** 85%

---

# 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Model Architecture](#-model-architecture)
- [Data Preparation](#-data-preparation)
- [Deployment](#-deployment-streamlit--supabase)
- [Performance](#-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📌 Project Overview

Bhasha Detector is a **lightweight transformer encoder–based model** designed to classify text into **22 Indic languages**, supporting both:

- **Native scripts** (e.g., हिंदी, தமிழ், বাংলা)  
- **Romanized forms** (e.g., "namaste", "vanakkam", "kem cho")  

Training is performed using a **two-phase pipeline**:

1. **Phase 1 – Triplet-Loss Embedding Model**  
   Learns language-discriminative embeddings using custom anchor–positive–negative triplets.
2. **Phase 2 – Cross-Entropy Classifier**  
   Uses embeddings to classify text into one of 22 languages.

A **Streamlit web app** deployed with **Supabase** provides real-time inference, batch CSV predictions, and user feedback logging for future DPO-style fine-tuning.

---

## 📂 Repository Structure

- **dataset-preparation**/ 
  - **training_dataset.ipynb** : This notebook was used for preparing the custom training dataset.
- **deployment**/ # Application codebase
  - **app.py** : Streamlit app entry point 
- **training**/ # Model training codebase
  - **main_singularity.py** : Main training script for both phases and evaluation
  - **model.py** : Custom transformer encoder architecture definition
  - **tokenizer.py** : Tokenizer and vocabulary creation utilities
  - **dataset.py** : Data loading and batching utilities
- **requirements.txt** : Python dependencies for the project
- **README.md** : Project documentation
- **LICENSE** : Project license information
- **.gitignore** : Files and directories to ignore in version control



---

## 🧠 Model Architecture

The custom transformer encoder (~40 MB) includes:

- Multi-head self-attention  
- Feed-forward encoder layers  
- Layer normalization & dropout  
- Mean pooling for sequence representation  
- Final 22-class classification head  

Optimized to balance **accuracy**, **speed**, and **model compactness**.

---

## 📑 Data Preparation

Using the **Bhasha-Abhijnaanam** dataset:

- **Anchor:** Native-script sentence  
- **Positive:** Romanized version of the same sentence  
- **Negative:** Romanized text from another language  

Preprocessing steps:

- Text cleaning & normalization using Indic NLP Library
- Transliteration handling  
- Tokenizer + vocabulary creation  

---

## 🚀 Deployment (Streamlit + Supabase)

The deployed app includes:

### ✔ Real-time inference  
Type text → instantly get predicted language.

### ✔ Batch CSV prediction  
Upload a CSV with a `text` column and track progress live.

### ✔ Supabase-backed feedback  
Users can submit corrections → stored for future **DPO fine-tuning**.

🔗 **Live Demo:**  
https://indic-language-identification.streamlit.app/

---

## 📈 Performance

| Metric          | Score |
|----------------|-------|
| **Accuracy**    | 87%   |
| **Macro-F1**    | 85%   |
| **Model Size**  | ~40 MB |

Shows strong performance on both **native-script** and **romanized** inputs.

---

## 📦 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Arka-Dutta-28/bhasha-detector
    cd bhasha-detector
    ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ``` 
3. **Run the Streamlit app:** 
   ```commandline
    streamlit run deployment/app.py
   ```
---

## 📘 Usage

- Use the Jupyter notebook (`.ipynb`) to generate and prepare the training data from the Bhasha-Abhijnaanam dataset.
- Use `main_singularity.py` for training and evaluation.  
  You can modify the hyperparameters inside the `__main__` section for hyperparameter tuning, model development, and tokenizer generation.
- The Streamlit app collects user input, loads the trained model and tokenizer, displays predicted language + softmax probabilities in real time, and logs incorrect predictions using Supabase.

---

## 🛠 Tech Stack

- **PyTorch** — Model architecture, training pipeline  
- **Streamlit** — Real-time user interface and deployment  
- **Supabase** — Feedback storage (for future DPO fine-tuning)  
- **Python** — Preprocessing, normalization, tokenization, backend logic  
- **NumPy / Pandas** — Data loading and manipulation  

---

## 📜 License

This project is open-source and available under the **MIT License**.

---
## 🙌 Acknowledgements

- **The authors of Bhasha-Abhijnaanam** - *Yash Madhani, Mitesh M. Khapra, and Anoop Kunchukuttan* - for releasing the dataset and benchmark that form the foundation of this project.
- **Streamlit** for enabling fast ML deployment  
- **Supabase** for providing simple and scalable backend support  

---


