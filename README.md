<div align="center">

# Causal Foundation Models with Continuous Treatments 

[![arxiv](https://img.shields.io/static/v1?label=arXiv&message=2408.16046&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2605.15133)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Layer6/CCPFN)

</div>

This is the repository for inference with CCPFN (**C**ontinuous **C**ausal **P**rior-**F**itted **N**etwork), a causal foundation model for use in domains with a continuous treatment variable. This is the inference repository; the full research reposistory (including training code and our prior) is forthcoming.

## Quick Start

To install from source (a PyPI package is forthcoming), run the following:
```bash
git clone https://github.com/layer6ai-labs/CCPFN-inference.git
cd CCPFN-inference
pip install -e .
```

Example notebooks on CCPFN usage, including individual treatment-response curve reconstruction tasks, can be found in the `notebooks` directory.

## Data Access

Various datasets are not included in the repository and must be downloaded separately.

### Warfarin

1. Download the data from [ClinPGx Downloads](https://www.clinpgx.org/downloads). Locate the **IWPC dataset (.xlsx)** under the "International Warfarin Pharmacogenetics Consortium (IWPC) Data" section.
2. Open the file and export the **"Subject Data"** sheet as a CSV file.
3. Update the file paths in `02_warfarin_example.ipynb` to point to your new CSV file.
