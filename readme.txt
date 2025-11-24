MSD-DDA: Multi-source Discriminant Dynamic Domain Adaptation for Cross-subject Motor Imagery EEG Recognition
**
This repository implements the MSD-DDA (Multi-source Discriminant Dynamic Domain Adaptation) model proposed in the paper Multi-source Discriminant Dynamic Domain Adaptation for Cross-subject Motor Imagery EEG Recognition, aiming to address the problem of poor generalization of motor imagery (MI) EEG recognition models across different subjects.
1. Model Introduction
To tackle the poor cross-subject generalization of motor imagery (MI) EEG recognition models, MSD-DDA proposes a multi-source discriminant dynamic domain adaptation solution:
Dynamically align global domain and local subdomain discrepancies to resolve the distribution mismatch between source and target domains;
Introduce Batch Nuclear Norm Maximization (BNM) to ensure the discriminability and diversity of target domain predictions;
Adopt a weighted ensemble prediction mechanism that automatically adjusts weights based on the similarity between source and target domains, suppressing negative transfer.
2. Experimental Results
Classification accuracy on 3 public MI EEG datasets (standardized names):
Dataset
Accuracy
BCI Competition IV Dataset 1
92.43% ± 5.05
BCI Competition IV Dataset 2a
79.24% ± 13.11
OpenBMI Dataset
71.96% ± 11.27

3. Environmental Dependencies
Python ≥ 3.7
PyTorch ≥ 1.8.0
Basic libraries: numpy, scipy, argparse
CUDA acceleration is recommended (e.g., NVIDIA GTX 3090/NVIDIA Tesla P40)
4. Quick Start
Data Preparation
Place the preprocessed EEG data (in .mat format) into the specified directory, including the data folder (storing EEG signals) and new_label (storing corresponding labels).
@article{GONG2025MSDDDA,
title={Multi-source Discriminant Dynamic Domain Adaptation for Cross-subject Motor Imagery EEG Recognition},
author={Yifan Gong, Kaiting Shi, Xiaolong Niu, Lijun Yang, Xiaohui Yang, Chen Zheng},
journal={IEEE Journal of Biomedical and Health Informatics},
year={2025}
}
