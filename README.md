
### Satellite Change Detection with TensorFlow (LEVIR-CD)

This project implements a deep learning model to detect changes in satellite imagery over time. Using the LEVIR-CD dataset, the model learns to identify man-made changes such as building constructions or demolitions by comparing two high-resolution images captured at different times.
<img width="1240" height="1440" alt="image" src="https://github.com/user-attachments/assets/cd619e5c-f40f-470e-a7db-50e9c361d03a" />

---

##  Features

- Siamese UNet architecture built in TensorFlow
- Pixel-level change detection on real satellite images
- Dataset preprocessing and visualization utilities
- Evaluation metrics: Precision, Recall, F1 Score
- Modular code structure for easy dataset or model replacement

---

##  Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
````

---

##  Project Structure

```
.
├── data/
│   ├── raw/                 # Unzipped LEVIR-CD dataset (A, B, label)
│   ├── processed/           # Resized image pairs and masks
│   └── scripts/
│       └── download_datasets.py
├── notebooks/               # (Optional) Jupyter notebooks
├── src/
│   ├── preprocessing.py     # Image loading, resizing
│   ├── model.py             # Siamese UNet model
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation metrics
├── requirements.txt
└── README.md
```

---

##  Dataset: LEVIR-CD

LEVIR-CD contains 637 pairs of high-resolution (1024×1024) images and change masks. It focuses on building change detection over time.
<img width="547" height="550" alt="image" src="https://github.com/user-attachments/assets/a6f2ac53-cde3-4839-af8c-18afb4377af3" />

To use it:

1. [Download LEVIR-CD from Google Drive or Baidu](https://justchenhao.github.io/LEVIR/)
2. Extract the folder and place the contents into:

```
data/raw/levir_cd/
├── A/
├── B/
└── label/
```

3. Alternatively, you can modify and run the script:

```bash
python data/scripts/download_datasets.py
```

---

##  Training the Model

```bash
python src/train.py
```

This will:

* Load the paired before/after images and corresponding masks
* Train a Siamese UNet model using binary crossentropy
* Save the model as `model_levir_cd.h5`

---

## Evaluation

To evaluate the trained model:

```bash
python src/evaluate.py
```

This computes:

* Precision
* Recall
* F1 Score

You can modify the script to print or save predictions as images.

---

##  Model Architecture

* **Input**: Two RGB images of shape (256×256×3)
* **Backbone**: Siamese UNet
* **Output**: Single binary mask (change vs no change)
* **Loss**: Binary crossentropy

You can swap in more complex models later (e.g., ResUNet, DeepLabV3).

---

##  Visualizing Predictions

Use a notebook or custom script to visualize outputs:

```python
from tensorflow.keras.models import load_model
model = load_model('model_levir_cd.h5')

pred = model.predict([X1_val, X2_val])[0]
plt.imshow(pred[:, :, 0] > 0.5, cmap='gray')
```

---

##  Metrics

* Binary Accuracy
* Precision
* Recall
* F1 Score
* (IoU and Dice score: coming soon)

---
<img width="550" height="283" alt="image" src="https://github.com/user-attachments/assets/88b8c421-0701-44a6-a97b-0315eb79dcce" />

##  Future Plans

* Add CRF and other post-processing options
* Support for other datasets (OSCD, CDD)
* Web UI for live change detection using Streamlit
* Add data augmentation and advanced loss functions (Dice, Tversky)

---

##  Citation

If you use the LEVIR-CD dataset, please cite:

> Chen, H., & Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. *arXiv preprint arXiv:2004.05502*

---

##  Maintainer


---


