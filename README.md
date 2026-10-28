<div align="center">

# 🛠️ Predictive Maintenance & Failure Prediction System

**Predict machine failures from sensor readings with a Random Forest pipeline and an interactive web app.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Data-AI4I%202020-blue)
![Status](https://img.shields.io/badge/Purpose-Educational%20%2F%20Portfolio-green)

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Try the App](#-try-the-app) · [Metrics](#-model-evaluation) · [FAQ](#-faq) · [Limitations](#%EF%B8%8F-limitations)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Try the App](#-try-the-app)
- [Model Evaluation](#-model-evaluation)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Configuration & Customization](#-configuration--customization)
- [Roadmap](#-roadmap)
- [FAQ](#-faq)
- [Limitations](#%EF%B8%8F-limitations)

---

## 🔍 Overview

Unplanned machine downtime is expensive. This project trains a classifier on the
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
to flag likely failures from six inputs, then serves the model through a simple web UI.

| | |
|---|---|
| **Task** | Binary classification (`Machine failure`: 0 / 1) |
| **Model** | `RandomForestClassifier` (200 trees, `class_weight="balanced"`) |
| **Inputs** | Machine type, air temp, process temp, rotational speed, torque, tool wear |
| **Outputs** | Failure prediction + estimated failure probability |
| **Interface** | Streamlit web app (`app.py`) |

---

## 🚀 Quick Start
## 🎮 Try the App

| Input | Default | Unit |
|---|---|---|
| Machine Type | `L` (options: L, M, H) | quality variant |
| Air Temperature | 298.5 | K |
| Process Temperature | 308.5 | K |
| Rotational Speed | 1500 | rpm |
| Torque | 40.0 | Nm |
| Tool Wear | 100 | min |

Click **Predict Machine Failure** to see a verdict (✅ no failure or ⚠️ potential failure) and the **Estimated Failure Probability**.

> Add a screenshot here, e.g. `![App demo](assets/demo.png)`.

---

## 📊 Model Evaluation

| Artifact | Location | Contents |
|---|---|---|
| Trained pipeline | `models/model.joblib` | Preprocessing + classifier |
| Metrics | `models/metrics.json` | ROC-AUC, average precision |
| Classification report | `reports/classification_report.txt` | Precision / recall / F1 |
| Confusion matrix | `reports/confusion_matrix.png` | Error breakdown |

Failures are rare in this dataset, so accuracy alone is misleading. Focus on recall, precision and average precision for the Failure class.

| Metric | Value |
|---|---|
| ROC-AUC | _TBD_ |
| Average precision | _TBD_ |
| Precision (Failure) | _TBD_ |
| Recall (Failure) | _TBD_ |
| F1 (Failure) | _TBD_ |

---

## 🗂️ Project Structure

```text
predictive-maintenance-ml/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── ai4i2020.csv
├── models/
│   ├── model.joblib
│   └── metrics.json
└── reports/
    ├── classification_report.txt
    └── confusion_matrix.png
```

---

## 📦 Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI), a synthetic dataset of 10,000 records.

1. Download it from the [UCI page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
2. Save the CSV as `data/ai4i2020.csv`.

---

## 🔧 Configuration & Customization

- **Change the model:** edit the `RandomForestClassifier(...)` block in `train.py`, then re-run `python train.py`.
- **Adjust the threshold:** in `app.py`, use `prediction = int(probability >= 0.35)` instead of `model.predict()` to catch more failures at the cost of more false alarms.
- **Add or remove features:** update the feature lists in `train.py` and the input widgets in `app.py`; column names must match exactly.

---

## 🗺️ Roadmap

- [x] Preprocessing + Random Forest pipeline
- [x] Evaluation metrics and confusion matrix
- [x] Interactive Streamlit app
- [ ] Add app screenshot / demo GIF
- [ ] Probability calibration
- [ ] Threshold tuning
- [ ] SHAP explanations in the app
- [ ] Cross-validation and model comparison
- [ ] Tests and CI
- [ ] Deploy to Streamlit Community Cloud

---

## ❓ FAQ

**"Dataset not found" when training?** Put `ai4i2020.csv` in `data/` and run `python train.py` from the project root.

**"Trained model not found" in the app?** Run `python train.py` first, or make sure `models/model.joblib` exists. Run the app from the project root.

**Is the probability a real-world risk?** No. It is the Random Forest's vote share, not a calibrated estimate, and the data is synthetic.

---

## ⚠️ Limitations

- The dataset is **synthetic**; results may not transfer to real machines.
- For **educational and portfolio purposes** only, not validated for industrial safety decisions.
- The displayed probability is **not calibrated**.
- Only point-in-time readings are used; there is no time-series history.

---

<div align="center">

Built by [Harish Mahesh Kumar](https://github.com/Harishmaheshkumar)

</div>
>
