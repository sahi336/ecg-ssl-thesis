# Label-Efficient and Uncertainty-Calibrated Self-Supervised Deep Learning for 12-Lead ECG Classification

Code accompanying an MSc thesis on self-supervised pre-training for 12-lead ECG
classification, with a focus on label efficiency, calibration, and robustness to
missing leads.

## Overview

A shared one-dimensional residual encoder is pre-trained on unlabelled ECG data using
three self-supervised objectives, then evaluated against a supervised baseline at 5%
and 10% label fractions. Predictions are calibrated with temperature scaling and the
calibrated model is stress-tested under lead removal.

Self-supervised methods compared:

- Masked signal reconstruction (zero-masking)
- SimCLR
- Contrastive Predictive Coding (CPC)

## Dataset

PTB-XL v1.0.3 — 21,799 clinical 12-lead recordings from 18,869 patients, labelled with
five diagnostic superclasses. Available from PhysioNet:
https://physionet.org/content/ptb-xl/

The official patient-separated fold structure is used, with fold 9 held out, so no
patient appears in more than one split.

## Notebooks

Run in order:

| Notebook | Purpose |
|---|---|
| `01_data_loading.ipynb` | Load PTB-XL records and diagnostic labels |
| `02_preprocessing.ipynb` | Normalisation and train/validation/test partitioning |
| `03_ssl_pretraining.ipynb` | Self-supervised pre-training of the shared encoder |
| `04_finetuning.ipynb` | Linear probing and end-to-end fine-tuning |
| `05_calibration.ipynb` | Temperature scaling and calibration metrics |
| `06_reduced_leads.ipynb` | Missing-lead stress test |

## Environment

Experiments were run on Google Colab (free tier) with an NVIDIA Tesla T4 GPU.

- Ubuntu 22.04 LTS
- Python 3.12
- PyTorch 2.11 (CUDA 12.8)
- NumPy, scikit-learn, SciPy, pandas
- netcal (calibration metrics)
- wfdb (ECG record reading)

Note that Colab does not pin library versions; the versions above reflect the
environment at the time of writing.

## Evaluation

All downstream results are reported as mean ± standard deviation over five random
seeds. The missing-lead stress test uses a single fixed seed with its own fitted
temperature. Expected Calibration Error is computed per class and averaged, as the
task is multi-label.

## Notes

Notebook 06 was run on CPU rather than GPU owing to quota limits on the free tier.
