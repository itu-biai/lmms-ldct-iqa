# LMM-IQA: Large Multimodal Model-Based Image Quality Assessment for Low-Dose CT

This repository contains Python scripts and data examples for the evaluation of large multimodal models (LMMs) in low-dose computed tomography (LDCT) image quality assessment. The project investigates how vision-language models perform in scoring LDCT images by comparing zero-shot, few-shot, metadata-guided, and error-feedback scenarios.

---

## Project Overview

The aim of this project is to assess whether general-purpose multimodal large language models can evaluate CT image quality similarly to radiologists. Each model predicts both a numerical image quality score and a textual explanation. The approach integrates contextual metadata (region and noise level) and uses an error-feedback mechanism to refine predictions.

The evaluation includes multiple models such as GPT-4o, Gemini 2.5 Pro, O3, Llama 4, Claude Sonnet 4, Qwen-VL-Max, and Grok 2.

---

## Directory Structure

```text
lmms_ldct_iqa/
│
├── AI Prediction File Examples/
│   ├── Gemini Scores Few Shot.txt
│   ├── GPT 4o Scores Few Shot.txt
│   ├── O3 Scores EF.txt
│   ├── O3 Scores Few Shot.txt
│   ├── O3-34 Scores Few Shot.txt
│   └── O3-Label-Noise Scores Few Shot.txt
│
├── Few-Shot/
│   ├── AI-FewShot-IQA-Score-Compare.py
│   ├── GPT-FewShot-API.py
│   ├── GPT-4omini-FewShot-API.py
│   ├── Meta-Llama4mav-FewShot-API.py
│   ├── Qwen-FewShot-API.py
│   ├── Gemini-FewShot-API.py
│   ├── Gemma3-FewShot-API.py
│   ├── O3-FewShot-API.py
│   ├── O3-34-API.py
│   ├── O3-Metadata-FewShot.py
│   └── O3-EF-FewShot.py
│
├── Few-Shot_10-Image/
│   ├── AI-FewShot-IQA-Score-Compare.py
│   ├── Claude-Sonnet4-FewShot.py
│   ├── Gemini-FewShot-API.py
│   ├── GPT-FewShot-API.py
│   ├── GPT-4omini-FewShot-API.py
│   ├── Gemma3-FewShot-API.py
│   ├── Grok2-FewShot-API.py
│   ├── Meta-Llama4mav-FewShot-API.py
│   ├── O3-FewShot-API.py
│   └── Qwen-FewShot-API.py
│
├── Zero-Shot/
│   ├── AI-ZeroShot-IQA-Score-Compare.py
│   ├── Claude-Sonnet4-ZeroShot.py
│   ├── Gemini-ZeroShot-API.py
│   ├── GPT-ZeroShot-API.py
│   ├── GPT-4omini-ZeroShot-API.py
│   ├── Gemma3-ZeroShot-API.py
│   ├── Grok2-ZeroShot-API.py
│   ├── Meta-Llama4mav-ZeroShot-API.py
│   ├── O3-ZeroShot-API.py
│   └── Qwen-ZeroShot-API.py
│
├── Noise File Examples/
│   ├── Noise Train Final.txt
│   └── Noise Test Final.txt
│
├── Radiolog Score File Examples/
│   ├── Radiolog Scores Few Shot Training.txt
│   └── Radiolog Scores Few Shot Testing.txt
│
└── Regions File Examples/
    ├── Region Labels Train.txt
    └── Region Labels Test.txt
```

---

## Key Features

- **Zero-Shot and Few-Shot Evaluation:** Assessing model performance with and without prior examples.
- **Metadata-Guided Prediction:** Using anatomical region and noise-level metadata to improve contextual understanding.
- **Error Feedback Mechanism:** Allowing iterative refinement of predictions based on prior results.
- **Model Comparison:** Evaluating GPT, Gemini, Llama, O3, Qwen, Claude, and Grok models under identical conditions.
- **Correlation Analysis:** Comparing AI-predicted scores with radiologist reference scores using PLCC, SROCC, and KROCC metrics.
- **Dataset Integration:** Structured examples for region labels, noise values, and human reference data for reproducibility.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/LMM-IQA.git
   cd LMM-IQA
   ```

2. **Create and activate a conda environment:**
   ```bash
   conda create -n lmmiqa python=3.10
   conda activate lmmiqa
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

All dependencies are listed in `requirements.txt`. Main packages include:

- `torch` (≥1.13.0)
- `torchvision` (≥0.14.0)
- `transformers` (≥4.39.0)
- `openai` (≥1.3.0)
- `numpy`, `pandas`, `scikit-image`, `opencv-python`, `matplotlib`, `tqdm`

---

## Evaluation Metrics

The evaluation is based on statistical correlation between radiologist scores and model predictions. Three metrics are used:

- **PLCC** (Pearson Linear Correlation Coefficient): Measures linear relationship between AI and radiologist scores.
- **SROCC** (Spearman Rank Correlation Coefficient): Evaluates rank-order consistency.
- **KROCC** (Kendall Rank Correlation Coefficient): Tests ordinal correlation and robustness to outliers.

**Overall Score:** Defined as `PLCC + SROCC + KROCC`, providing a unified performance index.

### Running Evaluations

The comparison scripts (e.g., `AI-FewShot-IQA-Score-Compare.py` and `AI-ZeroShot-IQA-Score-Compare.py`) automatically:

1. Read model prediction files from `AI Prediction File Examples/`
2. Load the corresponding radiologist reference files from `Radiolog Score File Examples/`
3. Match images by filename and compute PLCC, SROCC, and KROCC values4. Print a summary table and optionally export results as `.csv` for plotting

**Example usage:**

```bash
python Few-Shot/AI-FewShot-IQA-Score-Compare.py
```

To add new prediction sets, place them under `AI Prediction File Examples/` and reference them in the script.

---

## Citation

If you use this repository or methodology, please cite:

```bibtex
@article{celik2025lmm,
  title={LMM-IQA: Image Quality Assessment for Low-Dose CT Imaging},
  author={Celik, Kagan and Unal, Mehmet Ozan and Ertas, Metin and Yildirim, Isa},
  journal={arXiv preprint arXiv:2511.07298},
  year={2025}
}
```

---

## Contact

For questions and feedback, please open an issue in the GitHub repository or contact the authors directly.
