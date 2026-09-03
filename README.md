
<p align="center">
  <img src="assets/banner_transparent.png" alt="Banner" width="650">
</p>

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/license-GPL%20v3.0-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20618338.svg)](https://doi.org/10.5281/zenodo.20618338)
[![Version](https://img.shields.io/badge/version-v3.1-8A2BE2)](https://github.com/aurumz-rgb/RCT-Reviewer/releases)
[![Maintained](https://img.shields.io/badge/maintained-yes-success)](https://github.com/aurumz-rgb/RCT-Reviewer/graphs/commit-activity)
[![CI](https://github.com/RCT-Reviewer/RCT-Reviewer-Online/actions/workflows/ci.yml/badge.svg)](https://github.com/RCT-Reviewer/RCT-Reviewer-Online/actions/workflows/ci.yml)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)


**RCT-Reviewer** is a modernized, standalone version of [RobotReviewer](https://github.com/ijmarshall/robotreviewer), designed as a third-party reference tool for Risk of Bias assessment. It builds upon RobotReviewer’s original machine learning models trained on 12,808 randomized controlled trials (RCTs).

---

## ⚛️ Why use RCT-Reviewer?

RCT-Reviewer is designed as a **Third-Party Tiebreaker Reference** for systematic reviews. Standard guidelines require two independent human reviewers; when they disagree, this tool provides an instant, objective, and data-driven third opinion to resolve ties.

*   **Near-Human Accuracy**: The system achieves **71.0% accuracy** for Risk of Bias judgments, performing within **<8% of human expert consensus** (which stands at 78.3%) [1].

*   **Highly Precise Extraction**: In a randomized Cochrane user trial, the models demonstrated **87% Precision** and **90% Recall** for identifying the exact text snippets supporting the bias judgment [2].

*   **Validated Acceptance**: Real-world feasibility studies show that human reviewers accept the tool's judgments at a rate equal to that of their human peers (Risk Ratio 1.02) [3].

*   **Rigorous Methodology**: Developed by Marshall, Kuiper, and Wallace, the models were trained on **12,808 clinical trial PDFs** using "distant supervision" to ensure high-quality classification without prohibitive manual labeling costs [1,4].


---

## ⚙️ Validation

RCT-Reviewer has been independently validated against the original 2017 RobotReviewer implementation using a dedicated **four-tier validation harness**.

* **Predictive validity:** On 751 human-labelled Clinical Hedges records, RCT-Reviewer achieved **91.5% accuracy**, 94.1% sensitivity, 88.1% specificity, 0.925 F1 and 0.966 ROC AUC.
* **RobotReviewer fidelity:** The refactored RCT classification pipeline reproduces the executed original **bit-identically** — **751/751 records**, with max |Δ| = 0.0.
* **Risk-of-Bias fidelity:** **100% agreement** with the original across 6,018 document × domain comparisons (κ = 1.0), including identical sentence scores and vectorizer outputs.
* **SVM/CNN ablation:** The SVM-only pipeline achieves **92.5% decision agreement** with the original full SVM+CNN+publication-type ensemble; most of the difference is attributable to the legacy publication-type features.
* **PDF robustness:** **1,000/1,000 PDFs** parsed successfully, with a median processing time of 1.66 seconds per PDF.
* **External human validation (Tier E, Tian 2024, n=313 open-access trials):** concordance **60–76%** across the four RoB domains (κ 0.26/0.20/0.48/0.12) — within the range published for the original tool (κ 0.25–0.59, concordance 63–83%).
* **Human-reference control:** on identical PMC text, the original 2017 code agrees with RCT-Reviewer on **100.0%** of domain judgements; external fidelity vs the original's deposited publisher-PDF labels: **78.9%**.
* **Predictive validity detail:** PPV **91.1**, NPV **92.1**, Cohen's κ **0.826**, Brier score 0.067.
* **Parser robustness detail:** **12,060 pages** parsed, success 95% CI 99.6–100.0, max 7.0 s per PDF, parse time scales linearly (Pearson r = 0.93).

The complete methodology, reproducibility data, statistical results and validation outputs are available in the **[RCT-Reviewer Validation repository](https://github.com/RCT-Reviewer/Validation)**.

---

##  Architecture & Models

This project preserves the machine learning core of the original RobotReviewer while modernizing the infrastructure.

### The ML Pipeline
1.  **PDF Parsing**: Uses **PyMuPDF (fitz)** to extract text and **spaCy** to segment sentences. No Java/GROBID dependency required.
2.  **Feature Extraction**: Uses `HashingVectorizer` (scikit-learn) to convert text into high-dimensional sparse matrices.
3.  **Classification**:
    * MiniClassifier: A lightweight Linear SVM wrapper that loads pre-trained `.npz` weights.
    * **SVM-Only Pipeline**: CNN models have been removed due to TensorFlow compatibility issues on Python 3.11–3.12 and are not required for accurate predictions.

### Model Redistribution Note
This repository contains joblib-converted model artifacts originally developed in RobotReviewer. The old pickle model files were compressed and redistributed into `.joblib` format using the `convert_models.py` script and respective directories to ensure better compatibility and smaller file sizes for modern Python environments.


<img src="assets/huggingface.png" width="90" align="right" />

### Hugging Face Repository

The users can access the RCT-Reviewer Hugging Face Repository at:
[huggingface.co/Aurumz/RCT-Reviewer](https://huggingface.co/Aurumz/RCT-Reviewer)


<details>
<summary><b>📂 Project File Structure</b></summary>

```text
PROJECT ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│
├── 🚀 Entry point detected
│
├── 📁 File Structure
├── 📄 .dockerignore
├── 📄 .gitattributes
├── 📄 .gitignore
├── 📄 .zenodo.json
├── 📄 INFO.md
├── 📄 LFS push:pull guide.md
├── 📄 LICENSE.txt
├── 📄 README.md
├── 📄 convert_models.py
│
├── 📁 data/
│   ├── 📁 bias/
│   │   ├── 📄 bias_doc_level.npz
│   │   ├── 📄 bias_sent_level.npz
│   │   ├── 📄 bias_ab.npz
│   │   ├── 📄 bias_prob_clf.pck
│   │   ├── 📄 domain_clf.pck
│   │   ├── 📄 overall_clf.pck
│   │   ├── 📄 drugbank.pck
│   │   └── 📁 humans/
│   │       ├── 📄 AC.hdf5 / AC.json / AC.pickle
│   │       ├── 📄 BOA.hdf5 / BOA.json / BOA.pickle
│   │       ├── 📄 BPP.hdf5 / BPP.json / BPP.pickle
│   │       └── 📄 RSG.hdf5 / RSG.json / RSG.pickle
│   ├── 📁 pico/
│   │   ├── 📄 I_idf.npz / I_model.npz
│   │   ├── 📄 O_idf.npz / O_model.npz
│   │   ├── 📄 P_idf.npz / P_model.npz
│   ├── 📁 rct/
│   │   ├── 📄 rct_svm_weights.npz
│   │   └── ... (CNN artifacts retained but unused)
│   └── 📁 vocab/
│       └── 📄 embeddings.200d.trimmed.npz
│
├── 📁 rct_reviewer/
│   ├── 📄 __init__.py
│   ├── 📄 app.py (Joblib Mode)
│   ├── 📄 app1.py (Pickle Mode)
│   ├── 📄 app2.py (HF Hub Mode)
│   ├── 📄 config.py
│   ├── 📁 processors/ (bias_robot, pico_robot, rct_robot)
│   └── 📁 ui/ (streamlit components)
│
└── 📄 requirements.txt
```
</details>

---

## 🔄 Differences from Original RobotReviewer


| Feature | Original RobotReviewer (2017) | RCT-Reviewer |
| :--- | :--- | :--- |
| **Interface** | Flask + React | **Streamlit** (Pure Python) |
| **PDF Parsing** | GROBID (Requires Java/Docker) | **PyMuPDF** (Native Python) |
| **Task Queue** | Celery + RabbitMQ | **Synchronous** (Local execution) |
| **Data Models** | MultiDict | **Pydantic** |
| **Deployment** | Docker Compose | **Local Streamlit Run** |
| **ML Core** | SVM / CNN | **Same Weights** (SVM prioritized) |
| **Validated SVM-only vs Full Ensemble** | Full SVM+CNN+publication-type ensemble | 92.5% decision agreement |
| **Core Purpose** | Automated Risk of Bias assessment for RCTs | Modernized standalone implementation for automated Risk of Bias assessment |
| **Underlying ML Research** | Original ML models trained on 12,808 RCT PDFs | Preserves the same trained ML models and weights |
| **Risk of Bias Accuracy** | ~71.0% agreement accuracy vs expert consensus | Same expected predictive accuracy because the same SVM weights are used |
| **Supporting Text Precision** | ~87% precision for rationale extraction | Same extraction models retained |
| **Supporting Text Recall** | ~90% recall | Same extraction models retained |
| **Model Storage** | Pickle / HDF5 / NPZ | Joblib / NPZ / legacy compatibility modes |
| **Compatibility** | Compatible with Python 3.6 | Modernized for Python 3.12 |


### Note on SVM vs CNN

RCT-Reviewer uses a **Linear SVM-only pipeline** because the original CNN depends on obsolete TensorFlow/Keras infrastructure. Validation shows **92.5% agreement** with the original SVM+CNN+publication-type ensemble, with the majority of the difference attributable to the legacy publication-type features. The SVM-only pipeline preserves the original classifier's scores exactly while providing a modern, deterministic and reproducible implementation.


---

**This repository allows users to run RCT-Reviewer locally by downloading the required model files (`.joblib`, `.pickle`, `.pck`) using Git Large File Storage (LFS) or running directly via Hugging Face Hub integration.**

> **Online Deployment:** The default hosted version at [rct-reviewer.streamlit.app](https://rct-reviewer.streamlit.app) uses `app2.py` (Hugging Face Hub). The repository for the online deployment is [RCT-Reviewer/RCT-Reviewer-Online](https://github.com/RCT-Reviewer/RCT-Reviewer-Online).


---


## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.12**: This project is optimized for the latest Python version.
2.  **Git**: For cloning the repository.
3.  **Git LFS (Large File Storage)**: Required if you plan to run the offline modes (`app.py` or `app1.py`) to download pre-trained ML model weights.

> **⚠️ Note:** This project strictly requires **Python 3.12**. It has been tested and verified to work perfectly on this version. 

---

## 🛠️ Installation & Usage 

This repository contains the main model files along with the code required to run RCT-Reviewer. There are three ways to run this application locally:

1.  **app.py**: Uses `.joblib` and `.npz` files (Modernized local storage).
2.  **app1.py**: Uses `.pickle`, `.pck`, and `.npz` files (Legacy local storage).
3.  **app2.py**: Connects to the Hugging Face Hub repository [`Aurumz/RCT-Reviewer`] (No large local files needed).

[**Recommended to run app2.py over other .py to run latest version.**]

### General Setup (Required for all modes)

Clone the repository and set up the virtual environment:

```bash
git clone https://github.com/aurumz-rgb/RCT-Reviewer.git
cd RCT-Reviewer

# Create virtual environment
python3.12 -m venv .venv

# Activate it
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP Model (Required for all modes)
python -m spacy download en_core_web_sm
```

---

### Running Specific Modes

Choose one of the following methods to run the app.
[**Recommended to run app2.py over other .py to run latest version.**]

<details>
<summary><b>Mode 1: app.py (.joblib Local)</b></summary>

This version uses compressed `.joblib` and `.npz` files. It requires downloading model weights via Git LFS.

**1. Pull Model Weights:**
```bash
git lfs install
git lfs pull
```

**2. Run:**
```bash
python -m streamlit run rct_reviewer/app.py
```
</details>

<details>
<summary><b>Mode 2: app1.py (Legacy .pickle Local)</b></summary>

This version uses the original `.pickle`, `.pck`, and `.npz` files. It also requires Git LFS.

**1. Pull Model Weights:**
```bash
git lfs install
git lfs pull
```

**2. Run:**
```bash
python -m streamlit run rct_reviewer/app1.py
```
</details>

<details>
<summary><b>Mode 3: app2.py (Default mode / Hugging Face Hub)</b></summary>

This version fetches models directly from the Hugging Face Hub. **You do not need to run `git lfs pull`**.

**1. Run:**
```bash
python -m streamlit run rct_reviewer/app2.py
```
*Note: This requires an active internet connection to fetch models from `Aurumz/RCT-Reviewer` on Hugging Face.*
</details>


---

###  Reproduce the Validation

For independent verification of predictive validity, model fidelity, SVM/CNN ablation and PDF parsing robustness, see the **[RCT-Reviewer Validation repository](https://github.com/RCT-Reviewer/Validation)**.


---

## 🚨 Troubleshooting

**Q: I get `FileNotFoundError: data/pico/P_model.npz`**
**A:** You likely skipped the Git LFS step. Run `git lfs pull` to download the actual model weights (only required for `app.py` and `app1.py`).

**Q: I get `ModuleNotFoundError: No module named 'rct_reviewer'`**
**A:** Ensure you are running the command from the root directory, or set your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Q: `Can't find model 'en_core_web_sm'`**
**A:** You forgot to download the spaCy model. Run:
```bash
python -m spacy download en_core_web_sm
```

---

## References

1. Marshall IJ, Kuiper J, Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. Journal of the American Medical Informatics Association. 2016;23(1):193-201. [doi](http://dx.doi.org/10.1093/jamia/ocv044)

2. Soboczenski F, et al. Machine learning to help researchers evaluate biases in clinical trials: a prospective, randomized user study. BMC Medical Informatics and Decision Making. 2019;19(1):96. [doi](http://dx.doi.org/10.1186/s12911-019-0814-z)

3. Nussbaumer-Streit B, et al. Automating risk of bias assessment in systematic reviews: a real-time mixed methods comparison of human researchers to a machine learning system. BMC Medical Research Methodology. 2022;22:160. [doi](http://dx.doi.org/10.1186/s12874-022-01649-y)

4. Marshall I, Kuiper J, Wallace B. Automating Risk of Bias Assessment for Clinical Trials. IEEE Journal of Biomedical and Health Informatics. 2015;19(4):1406-1412. [doi](http://dx.doi.org/10.1109/JBHI.2015.2431314)



---

## Contributing  
<img src="assets/main_logo_zoomed.png" width="110" align="right" />

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to suggest additions or changes.

<a href="https://github.com/aurumz-rgb/RCT-Reviewer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aurumz-rgb/RCT-Reviewer" />
</a>

---

##  Acknowledgements

RCT-Reviewer is a modernized version of the original [RobotReviewer](https://github.com/ijmarshall/robotreviewer). I extend my sincere gratitude to the original authors: **Iain J. Marshall, Joël Kuiper, Edward Banner, and Byron C. Wallace** for their foundational work in biomedical NLP and for releasing the project as open-source.

I would also like to thank all contributors and collaborators involved in the RobotReviewer ecosystem, including the Cochrane Crowd and the research teams at UPenn, Northeastern, and UCL, whose efforts in data collection and model development made this tool possible.

Additionally, I would like to acknowledge the use of [RikaiCode](https://rikaicode.github.io) (Code Repository Context Generator) and [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7), which were invaluable in analyzing and understanding the complex logic of the original [RobotReviewer](https://github.com/ijmarshall/robotreviewer) codebase, as well as assisting in the development and modernization of RobotReviewer.



---


## 📌 Citation

If you use this software in your research, please cite both RCT-Reviewer and the original RobotReviewer paper.

**RCT-Reviewer:**

Sahu, V. (2026). RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs). Zenodo. https://doi.org/10.5281/zenodo.20618338

```bibtex
@software{RCT-Reviewer,
  author    = {Sahu, V.},
  title     = {RCT-Reviewer: A Modernized, Standalone Tool for Automated Analysis of Clinical Trials (RCTs)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20618338},
  url       = {https://doi.org/10.5281/zenodo.20618338}
}
```

**Original RobotReviewer:**

Marshall IJ, Kuiper J, Banner E, Wallace BC. “Automating Biomedical Evidence Synthesis: RobotReviewer.” Proceedings of the Conference of the Association for Computational Linguistics (ACL). 2017 (July): 7–12.

```bibtex
@article{RobotReviewer2017,
  title    = "Automating Biomedical Evidence Synthesis: {RobotReviewer}",
  author   = "Marshall, Iain J and Kuiper, Jo{\"e}l and Banner, Edward and Wallace, Byron C",
  journal  = "Proceedings of the Conference of the Association for Computational Linguistics (ACL)",
  volume   = 2017,
  pages    = "7--12",
  month    = jul,
  year     = 2017,
}
```

---

[![GNU GPL v3 License](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)](https://www.gnu.org/licenses/gpl-3.0.en.html)


This project is a derivative work of [RobotReviewer](https://github.com/ijmarshall/robotreviewer) and is distributed under the GNU GENERAL PUBLIC LICENSE v3.0.
