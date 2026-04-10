# Face Recognition — From Eigenfaces to Deep Learning

A face recognition system that has evolved over **10 years** — from a classical MATLAB PCA/Eigenfaces implementation (2014) to a modern PyTorch deep-learning pipeline (2026).

---

## The Journey

### 2014 — Classical Approach (MATLAB)

The original implementation used **Principal Component Analysis (PCA)** and **Eigenfaces** to recognise faces. Images were flattened into 1-D vectors, projected onto a low-dimensional eigenspace derived from the training set's covariance matrix, and compared using Euclidean distance. It was built with MATLAB's GUI dialog boxes and hardcoded Windows paths.

The legacy code is preserved in [`legacy/matlab/`](legacy/matlab/) as a snapshot of where this project started.

### 2026 — Deep Learning Approach (PyTorch)

The modern version replaces hand-crafted linear algebra with a **ResNet-18 convolutional neural network** trained using **Triplet Margin Loss**. Instead of comparing raw pixel projections, the model learns a 128-dimensional embedding space where faces of the same person cluster together and faces of different people are pushed apart. Recognition is performed via cosine similarity.

**What changed in 10 years:**

| Aspect | 2014 (MATLAB) | 2026 (PyTorch) |
|---|---|---|
| Feature extraction | PCA / Eigenfaces | ResNet-18 CNN |
| Loss function | N/A (unsupervised) | Triplet Margin Loss |
| Comparison metric | Euclidean distance on eigenspace | Cosine similarity on learned embeddings |
| Data augmentation | None | Rotation, flip, colour jitter, grayscale |
| Hardware | CPU only | GPU / Apple Silicon / CPU |
| Portability | Windows-only MATLAB paths | Cross-platform Python |

---

## Project Structure

```
Facial-Recognition/
├── legacy/
│   └── matlab/               # Original 2014 MATLAB implementation
│       ├── facerecog.m       # PCA eigenface computation & matching
│       ├── face_recognition.m # Entry-point GUI script
│       └── license.txt
├── src/
│   ├── model.py              # ResNet-18 embedding network
│   ├── dataset.py            # Triplet dataset & augmentation pipeline
│   ├── train.py              # Training loop with triplet loss
│   └── recognize.py          # Inference & gallery matching
├── data/                     # Training / gallery images (not committed)
├── models/                   # Saved model weights (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/Facial-Recognition.git
cd Facial-Recognition
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Prepare Your Data

Organise face images into one sub-folder per identity:

```
data/train/
  person_a/
    001.jpg
    002.jpg
  person_b/
    001.jpg
    002.jpg
```

Each identity needs at least **2 images** for triplet training.

### Train

```bash
python -m src.train \
    --data_dir data/train \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --embedding_dim 128
```

The best model weights are saved to `models/face_embedding.pth`.

### Recognise a Face

Set up a gallery directory with the same folder-per-identity structure, then query:

```bash
python -m src.recognize \
    --model_path models/face_embedding.pth \
    --gallery_dir data/gallery \
    --query path/to/test_face.jpg \
    --top_k 3
```

The script prints the top-k closest identities ranked by cosine similarity.

---

## How It Works

### Training — Triplet Loss

Each training batch samples triplets of **(anchor, positive, negative)**:
- **Anchor** and **positive** are different images of the **same** person.
- **Negative** is an image of a **different** person.

The loss pushes the anchor closer to the positive and further from the negative by at least a configurable margin.

### Inference — Cosine Similarity

1. Every gallery image is passed through the model to produce a 128-D embedding.
2. The query image is embedded the same way.
3. Cosine similarity is computed between the query and every gallery embedding.
4. The identity with the highest similarity above the threshold is returned.

---

## License

The original MATLAB code is provided under a BSD-2-Clause license (see [`legacy/matlab/license.txt`](legacy/matlab/license.txt)).

The modern PyTorch implementation is released under the [MIT License](LICENSE).
