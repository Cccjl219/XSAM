# Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation

This is the official code for our ICLR 2026 paper **"[Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation
](https://openreview.net/forum?id=qTRqmMOOrH&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))"**.

## Getting Started

To reproduce our results, please follow the steps below.

### 1. Install Dependencies

First, make sure your environment has the required Python packages. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Run Training
Then, you can run train.py using commands such as the following:
```bash
For SGD+:
python train.py --model='resnet18' --datasets='cifar100' --rho=0.15 --alpha=0.0
---
For SAM:
python train.py --model='resnet18' --datasets='cifar100' --rho=0.15 --alpha=1.0
---
For XSAM:
python train.py --model='resnet18' --datasets='cifar100' --rho=0.15 --rho_max=0.3 --is_dynamic

```


