# Multiple Disease Risk Prediction System

**Final-year B.Tech capstone | DIT University | 2026**

A prototype **Streamlit** application bringing together three independent machine-learning modules for **heart disease, diabetes, and chronic kidney disease (CKD)** risk classification. The project compares classification algorithms on public datasets and presents model predictions through a form-based interface.

> **Educational prototype only.** This is not a medical device, clinical diagnostic tool, or validated substitute for professional medical advice. Reported test-set results do not establish real-world clinical performance.

## Project overview

- **Three prediction modules:** heart disease, diabetes, CKD
- **ML methods:** Logistic Regression, Support Vector Machine, Random Forest, XGBoost; the presentation also reports Decision Tree comparisons
- **Preprocessing:** missing-value handling, duplicate removal, scaling, correlation-informed feature selection, train/test splitting
- **Evaluation:** accuracy, precision, recall, F1 score, ROC-AUC
- **Interface:** Python + Streamlit; separate forms and results for each disease

## Reported model comparison

These numbers are **transcribed from the capstone presentation**, not independently reproduced by this repository. They should be verified against the original notebooks, dataset versions, preprocessing, and splits before being presented as reproducible benchmarks.

| Algorithm | Heart accuracy | Heart recall | Diabetes accuracy | Diabetes recall | Kidney accuracy | Kidney recall |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8525 | 0.8667 | 0.7338 | 0.8889 | 1.0000 | 1.0000 |
| Logistic Regression | 0.7705 | 0.8519 | 0.7468 | 0.8704 | 0.9750 | 1.0000 |
| SVM | 0.7869 | 0.8519 | 0.7273 | 0.8889 | 1.0000 | 1.0000 |
| XGBoost | 0.7869 | 0.8370 | 0.7338 | 0.8889 | 0.9500 | 0.9400 |
| Decision Tree | 0.7377 | 0.7377 | 0.7662 | 0.8148 | 0.9875 | 1.0000 |

**Interpretation note:** The slides state that model selection prioritised recall, and identify Random Forest, Logistic Regression and XGBoost as selected models, but do not unambiguously map each final model to each disease. The exact saved model per module must be checked in the original code. A reported 1.0 test score on a small dataset is not evidence of clinical reliability; inspect class balance, duplicates, leakage and evaluation protocol.

## Data sources

| Module | Source | Records reported in presentation | Features reported |
|---|---|---:|---:|
| Heart disease | [Kaggle: johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) | 1,025 | 14 |
| Diabetes | [Kaggle: iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) | 100,000 | 8 |
| CKD | [UCI: Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | 400 | 24 |

**Dataset discrepancy to resolve:** The presentation cites the 100,000-row Kaggle diabetes dataset but lists glucose, insulin, pregnancies, skin thickness and pedigree function, which are associated with a different commonly used diabetes dataset. Check the original CSV and training notebook to establish which diabetes dataset was actually used. Do not upload a substitute and claim it was the training data.

See [`data/README.md`](data/README.md) for provenance, privacy and redistribution guidance.

## Architecture

```text
Streamlit UI → disease selection → feature inputs
           → preprocessing → disease-specific trained model
           → risk classification → result display
```

## Repository status

**Documentation starter — not yet runnable.** This repository contains a structured README, dataset references, and documentation templates. The original `app.py`, preprocessing/training scripts, fitted model artifacts, exact environment versions and non-sensitive screenshots still need to be added from the original project. **Do not claim a live demo or reproducible training until those files are tested.**

### Planned project layout

```text
multiple-disease-risk-prediction/
├── README.md
├── .gitignore
├── requirements.txt            # fill with verified dependencies
├── src/                        # original app and training code
├── models/                     # fitted artifacts, only if safe to share
├── data/
│   ├── README.md
│   └── raw/                     # data excluded from Git by default
├── notebooks/                   # original experimentation notebooks
├── assets/screenshots/         # redacted application screenshots
└── docs/
    ├── methodology.md
    └── reproducibility-checklist.md
```

## How to run (after source code is added)

1. Add and review original source code; identify its actual Streamlit entry point.
2. Create a virtual environment: `python -m venv .venv`.
3. Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux).
4. Install verified dependencies: `pip install -r requirements.txt`.
5. Download the **exact** dataset versions used; follow `data/README.md`.
6. Run the original training script if models must be generated, then run `streamlit run src/app.py` **only if that is the actual entry-point path**.

## Limitations and responsible use

- Educational proof of concept, not clinically validated.
- Public medical datasets may be unrepresentative of intended populations.
- Class imbalance, data leakage and dataset shifts can inflate apparent performance.
- No real patient data, personal identifiers or secrets should be committed.
- Do not load untrusted pickle/model files: deserialization can execute code.

## Credits

**Team capstone:** Abhyudita Pandey, Nishka Sharma and Raghav Sharma (Group BTCSE-100). **Project advisor:** Dr. Biswadip Bandyopadhyay, DIT University. Contributions to code and documentation should be attributed accurately after checking with the team.

## Sources

- Final-year capstone report (2026), *Multiple Disease Risk Prediction System*
- End-term capstone presentation (2026), *Multiple Disease Risk Prediction System*

## License

No software license has been selected yet. Confirm ownership and agreement of all contributors before choosing one; dataset licenses are separate.
