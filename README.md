# Handwriting Author Recognition using a ResNet-18 CNN

This project implements a custom ResNet-18 convolutional neural network (CNN) in TensorFlow/Keras to identify the author of a given sample of handwritten text. The model is trained and evaluated on the well-known **IAM Handwriting Database** using K-Fold cross-validation to ensure robust performance.

This repository provides the Jupyter Notebook to replicate the training process, from data preprocessing to model evaluation.

---

## Features

* **Model**: A ResNet-18 architecture built from scratch.
* **Dataset**: The public IAM Handwriting Database.
* **Preprocessing**: Images are loaded, converted to grayscale, resized to 64x64, and normalized. Author labels are mapped from the dataset's metadata.
* **Training**: Implements K-Fold cross-validation (10 folds) for robust training and to prevent evaluation bias.
* **Data Augmentation**: Uses `ImageDataGenerator` to perform on-the-fly augmentation (zoom, flips, rotation), making the model more robust and preventing overfitting.
* **Evaluation**: Provides detailed accuracy and loss metrics for each fold and visualizes the training history with Matplotlib.

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

> **Note**: A GPU is highly recommended for reasonable training times.

### 1. Prerequisites

First, ensure you have Python 3.8+ installed. Then, install the required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn jupyterlab
````

### 2\. Download the Dataset

The model is trained on the IAM Handwriting Database. You must download the necessary files from the official source:

  * **Official Website**: [IAM Handwriting Database Download Page](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)

> You will need to register for free to get access. From the download page, get the following files:

1.  `sentences.tgz`
2.  `forms_BA.txt` (This file is typically derived from the `forms.txt` in the "ASCII" section. Ensure you have a file that maps form IDs to writer IDs).

### 3\. Set Up the Project Directory

Create a main folder for the project and organize your files as follows. This structure is required for the notebook's file paths to work correctly.

```plaintext
handwriting-recognition-project/
├── Handwriting_Author_Recognition.ipynb
├── resnets_utils.py
└── data/
    ├── sentences.tgz
    └── forms_BA.txt
```

### 4\. Prepare the Notebook for Local Execution

The original notebook was designed for Google Colab. You must modify it to run locally.

1.  **Open the Notebook**: Launch Jupyter Lab (`jupyter lab` in your terminal) and open `Handwriting_Author_Recognition.ipynb`.
2.  **Remove Google Drive Code**: Delete the cells containing `google.colab` and `drive.mount()`.
3.  **Update File Paths**: Find and replace all hardcoded `/content/...` paths with relative local paths. Key changes include:
      * **Data Extraction**: Change `!tar -xvf /content/drive/MyDrive/sentences.tgz -C "/content/images"` to:
        ```python
        # Create a directory for the extracted images
        !mkdir -p images
        # Extract the archive into it
        !tar -xvf data/sentences.tgz -C "images/"
        ```
      * **Forms File Path**: Change `forms_file_info = "/content/drive/MyDrive/forms_BA.txt"` to:
        ```python
        forms_file_info = "data/forms_BA.txt"
        ```
      * **Image Copying Paths**: In the cell that copies files to a temporary directory, update the paths:
        ```python
        temp_sentences_path = "temp_sentences"
        original_sentences_path = "images/**/**/*.png" # Check if extraction creates a 'sentences' subfolder
        ```

### 5\. Run the Project

Execute the cells in the Jupyter Notebook sequentially. The script will preprocess the data, build the model, and begin the K-Fold training process.

-----

## Model Performance

The model was trained for 80 epochs per fold across 10 folds. The evaluation on the test sets for each fold yielded an average accuracy of approximately **60%**. The training history, including accuracy and loss curves for each fold, is plotted at the end of the notebook execution.

```
```