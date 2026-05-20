# Protocol-Aware EEG/BCI Decoding Pipelines

This repository serves as the front page for two EEG/BCI projects focused on **protocol-aware evaluation**, **leakage-controlled preprocessing**, and **generalization analysis**.

Rather than presenting EEG deep learning as a simple model-reproduction task, these projects aim to build reusable research pipelines for evaluating how EEG decoding models behave under more realistic experimental settings.

The current portfolio includes two complementary directions:

```text
1. Motor Imagery BCI Decoding
2. EEG-based Emotion Recognition / Affective BCI
```

Together, they reflect a broader research interest in computational neuroscience, EEG signal processing, brain-computer interfaces, and reliable deep learning evaluation for neural data.

---

## Project Overview

### Project 1: Conformer-based EEG/BCI Decoding Pipeline for Motor Imagery

This project develops a modular EEG/BCI decoding pipeline for four-class motor imagery classification using a convolutional Transformer architecture.

The project was initially motivated by EEG-Conformer, but the upgraded version moves beyond simple reproduction. It focuses on building a cleaner EEG decoding workflow, including pure motor imagery protocol refinement, leakage-controlled preprocessing, training-only data augmentation, Leave-One-Subject-Out cross-subject evaluation, training/validation curve analysis, and small-sample overfitting analysis.

The task is four-class motor imagery EEG decoding:

```text
0 = left fist imagery
1 = right fist imagery
2 = both fists imagery
3 = both feet imagery
```

The project uses the PhysioNet EEG Motor Movement/Imagery Dataset, also known as EEGMMIDB. The protocol is refined to pure motor imagery runs:

```text
Runs: 4, 6, 8, 10, 12, 14
```

This avoids mixing overt executed movement trials with imagined movement trials and makes the decoding task more consistent with motor imagery BCI research.

---

### Project 2: Protocol-Aware EEG Emotion Recognition Pipeline for Affective BCI

This project develops a protocol-aware EEG emotion recognition evaluation pipeline for affective BCI research.

The project was initially inspired by the LibEER benchmark for EEG-based emotion recognition, but it is not limited to a single-model reproduction. It focuses on leakage-aware preprocessing, baseline benchmarking, repeated validation, and transparent interpretation of model performance.

The key motivation is that extremely high accuracy under random-split or subject-dependent settings may not necessarily indicate true cross-subject emotion decoding ability. Therefore, this project emphasizes evaluation protocol analysis rather than simply reporting the highest test accuracy.

The current version uses the Kaggle EEG Brainwave Dataset: Feeling Emotions, which provides pre-extracted EEG feature vectors and emotion labels. Since this dataset does not provide subject identifiers, true Leave-One-Subject-Out validation is not claimed in the current version.

---

## Why These Two Projects Are Combined

These two projects address different EEG/BCI tasks but share the same methodological focus.

| Dimension | Motor Imagery BCI | Affective BCI |
|---|---|---|
| Main task | Motor imagery decoding | Emotion recognition |
| Dataset | PhysioNet EEGMMIDB | Kaggle Feeling Emotions |
| Model focus | EEG-Conformer / Transformer-based decoding | 1D-CNN and classical baselines |
| Evaluation focus | Cross-subject LOSO evaluation | Protocol-aware within-dataset evaluation |
| Core issue | Small-sample overfitting and cross-subject generalization | Random-split inflation and potential leakage |
| Research value | Tests whether Transformer-based EEG decoding generalizes to unseen subjects | Tests whether high EEG emotion accuracy is reliable under cleaner protocols |

The common goal is to build EEG pipelines that are not only runnable, but also scientifically interpretable.

---

## Unified Research Motivation

EEG decoding is challenging because EEG signals are noisy, non-stationary, subject-specific, and often limited in sample size.

In this context, reporting a single high accuracy number is often insufficient. A more reliable EEG/BCI project should also consider:

- Whether the task protocol is clearly defined
- Whether train/test leakage is avoided
- Whether evaluation is stable across folds or subjects
- Whether the model generalizes beyond the training distribution
- Whether high performance is caused by real neural patterns or protocol artifacts
- Whether failure cases and overfitting behavior are analyzed

These two projects were designed around this principle.

---

## Key Contributions

### 1. Framework

Built modular EEG/BCI pipelines for two representative decoding tasks:

```text
Motor Imagery BCI
EEG-based Emotion Recognition
```

Both pipelines are designed to move beyond isolated reproduction notebooks and toward reusable EEG model evaluation workflows.

---

### 2. Data

Refined EEG data handling for task-specific evaluation.

For motor imagery decoding, the EEGMMIDB protocol is refined to pure motor imagery runs:

```text
4, 6, 8, 10, 12, 14
```

For EEG emotion recognition, the Kaggle Feeling Emotions dataset is handled as a pre-extracted feature dataset, with its subject-identifier limitation explicitly documented.

---

### 3. Training

Implemented leakage-controlled training workflows.

Common design choices include:

- Train/test split before normalization
- Standardization fitted only on training data
- Training-only augmentation where applicable
- Validation-based early stopping
- Dropout regularization
- Reduced training epochs for pilot experiments
- Baseline comparison instead of single-model reporting

---

### 4. Evaluation

Designed evaluation protocols according to dataset availability.

For the motor imagery project:

```text
Leave-One-Subject-Out cross-subject evaluation
```

For the EEG emotion recognition project:

```text
Stratified hold-out validation
Repeated hold-out validation
Stratified K-fold validation
Label-permutation sanity check
```

Evaluation metrics include:

- Accuracy
- Balanced accuracy
- Macro-F1
- Weighted-F1
- Confusion matrix
- Mean ± standard deviation
- Training/validation curve analysis

---

### 5. Insight

Both projects emphasize interpretation rather than only performance.

The motor imagery project examines whether Transformer-based EEG decoding is stable under small-sample cross-subject settings.

The emotion recognition project examines whether near-perfect random-split accuracy can be trusted when subject identifiers are missing.

The overall insight is:

> In EEG/BCI research, evaluation protocol and leakage control are as important as model architecture.

---

## Repository Structure

A suggested combined structure is:

```text
.
├── README.md
│
├── motor_imagery_conformer/
│   ├── README.md
│   ├── EEG_BCI_Conformer_Pipeline_LOSO.ipynb
│   ├── requirements.txt
│   ├── results/
│   ├── figures/
│   └── docs/
│
├── emotion_recognition_pipeline/
│   ├── README.md
│   ├── eeg_emotion_evaluation_pipeline.ipynb
│   ├── requirements.txt
│   ├── RESULTS.md
│   ├── outputs/
│   └── docs/
│       └── protocol_analysis.md
│
└── docs/
    └── research_summary.md
```

If the two projects remain in separate GitHub repositories, this README can still be used as a front-page portfolio README that links to both repositories.

---

## How to Run

### Motor Imagery BCI Project

Open the notebook:

```text
EEG_BCI_Conformer_Pipeline_LOSO.ipynb
```

In Google Colab:

```text
Runtime → Change runtime type → GPU
```

For quick testing:

```python
RUN_FOLDS = 3
```

For a larger pilot run:

```python
RUN_FOLDS = 10
```

For full LOSO evaluation:

```python
RUN_FOLDS = None
```

---

### EEG Emotion Recognition Project

Open the notebook:

```text
eeg_emotion_evaluation_pipeline.ipynb
```

The notebook automatically:

1. Installs dependencies
2. Downloads the Kaggle dataset through `kagglehub`
3. Loads EEG feature data
4. Runs baseline benchmarking
5. Exports evaluation results

Example experiments:

```python
summary_dummy, results_dummy = run_experiment(
    model="dummy",
    protocol="kfold",
    epochs=1,
)

summary_logreg, results_logreg = run_experiment(
    model="logreg",
    protocol="kfold",
    epochs=1,
)

summary_cnn, results_cnn = run_experiment(
    model="cnn",
    protocol="kfold",
    epochs=40,
)
```

Optional sanity check:

```python
summary_perm, results_perm = run_experiment(
    model="cnn",
    protocol="kfold",
    epochs=20,
    permutation_sanity_check=True,
)
```

---

## Current Status

### Motor Imagery BCI

Current implementation includes:

- Pure motor imagery trial extraction
- Four-class label assignment
- Leakage-controlled preprocessing
- Training-only augmentation
- EEG-Conformer training
- LOSO evaluation
- Training/validation curve analysis
- Pilot fold control for runtime management

Pilot LOSO evaluation can be completed first before running the full 30-subject setup.

---

### EEG Emotion Recognition

Current implementation includes:

- Automatic Kaggle dataset download
- Leakage-aware standardization
- Dummy / Logistic Regression / 1D-CNN baselines
- Hold-out / repeated hold-out / K-fold evaluation
- Accuracy, balanced accuracy, macro-F1, weighted-F1
- Confusion matrix export
- Label-permutation sanity check
- Explicit discussion of subject-identifier limitations

Numerical results are currently under update and should be filled after running the standardized Colab pipeline.

---

## Notes on Interpretation

These projects do not aim to claim that one model architecture is universally superior.

Instead, they aim to answer more careful research questions:

```text
Can the model generalize to unseen subjects?
Is the evaluation protocol clean?
Is the reported accuracy stable?
Is there possible leakage?
Does the training curve indicate overfitting?
Are high results scientifically interpretable?
```

Low or unstable performance can still be informative, especially in cross-subject EEG decoding, where generalization is a known challenge.

---

## Future Work

Future improvements include:

- Run full 30-subject LOSO evaluation for motor imagery decoding
- Add CNN baselines such as EEGNet and ShallowConvNet
- Conduct Transformer-depth and attention-head ablation studies
- Add subject-adaptive fine-tuning
- Explore domain adaptation for cross-subject EEG decoding
- Migrate the emotion recognition pipeline to SEED, DEAP, or FACED
- Implement LOSO validation for affective BCI datasets with subject identifiers
- Add graph-based EEG models such as DGCNN
- Extend both pipelines toward pseudo-online BCI feedback

---

## Resume Summary

This combined project can be summarized as:

> Developed protocol-aware EEG/BCI decoding pipelines for motor imagery and affective computing, focusing on leakage-controlled preprocessing, baseline benchmarking, cross-subject/generalization analysis, and transparent evaluation of deep learning models for neural data.

---

## Keywords

```text
EEG
BCI
Motor Imagery
Affective Computing
Emotion Recognition
EEG-Conformer
Transformer
1D-CNN
PyTorch
MNE-Python
Scikit-learn
Cross-subject Evaluation
Leave-One-Subject-Out
LOSO
Benchmarking
Protocol-aware Evaluation
Leakage Control
Small-sample Overfitting
Computational Neuroscience
```
