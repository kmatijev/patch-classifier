# Mitosis Detection — Domain Shift Study

CNN-based pipeline for detecting mitotic figures in whole slide images (WSI), built on the [MIDOG 2021](https://midog2021.grand-challenge.org/) dataset. The main focus is **domain shift**: how well a model trained on one scanner generalises to others.

---

MIDOG 2021 annotations (`MIDOG.json`, COCO format) across three scanners: **Hamamatsu XR**, **Hamamatsu S360**, and **Aperio CS**. Raw TIFF images must be downloaded separately from the challenge.

---

Models are trained for each of 3 scanners × 4 augmentation strategies = **12 combinations**.

Augmentation strategies: `standard`, `medium`, `strong`, `histology` (H&E stain simulation).

---

## Installation

Python 3.9+, GPU recommended.

```bash
pip install torch torchvision opencv-python-headless scikit-learn pandas numpy matplotlib seaborn tqdm pillow
```