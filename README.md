# MSD-DDA: Multi-source Discriminant Dynamic Domain Adaptation for Cross-subject Motor Imagery EEG Recognition

This repository provides the official implementation of **MSD-DDA (Multi-source Discriminant Dynamic Domain Adaptation)** proposed in the paper:

> *Multi-source Discriminant Dynamic Domain Adaptation for Cross-subject Motor Imagery EEG Recognition*  
> Yifan Gong, Kaiting Shi, Xiaolong Niu, Lijun Yang, Xiaohui Yang, Chen Zheng, IEEE Journal of Biomedical and Health Informatics, 2025.

MSD-DDA aims to address the poor cross-subject generalization of motor imagery (MI) EEG recognition models.

---

## 1. Model Introduction

To tackle the cross-subject distribution shift in MI-EEG, **MSD-DDA** proposes a **multi-source discriminant dynamic domain adaptation** framework with three key components:

- **Dynamic global & local alignment**  
  Dynamically aligns both global domain discrepancy and local subdomain discrepancy to reduce distribution mismatch between multiple source domains and the target domain.

- **Batch Nuclear-norm Maximization (BNM)**  
  Introduces BNM to encourage **discriminability** and **diversity** of target-domain predictions, mitigating prediction collapse.

- **Weighted ensemble prediction**  
  Adopts a similarity-aware weighted ensemble strategy that automatically adjusts the contribution of each source domain according to its similarity to the target domain, thereby suppressing negative transfer.

---

## 2. Experimental Results

MSD-DDA is evaluated on three public MI-EEG datasets, achieving the following cross-subject classification accuracies:

| Dataset                         | Accuracy (mean ± std) |
|---------------------------------|------------------------|
| BCI Competition IV Dataset 1    | 92.43% ± 5.05         |
| BCI Competition IV Dataset 2a   | 79.24% ± 13.11        |
| OpenBMI Dataset                 | 71.96% ± 11.27        |

---

## 3. Environment Dependencies

- Python ≥ 3.7  
- PyTorch ≥ 1.8.0  
- Other Python packages: `numpy`, `scipy`, `argparse` (and common scientific Python libraries)

For best performance, **CUDA acceleration** is recommended, e.g.:

- NVIDIA GTX 3090 / NVIDIA Tesla P40 or similar
- PyTorch with CUDA support installed

---

## 4. Quick Start

### 4.1 Data Preparation

1. Preprocess the MI-EEG signals following the procedures described in the paper.  
2. Save the preprocessed data in `.mat` format.  
3. Organize the data into the designated directory structure, for example:

   ```text
   data_root/
     ├── data/        # EEG signals (.mat)
     └── new_label/   # Corresponding labels (.mat or .npy)

