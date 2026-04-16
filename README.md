# EEG Research Projects

[![GitHub Repo](https://img.shields.io/badge/GitHub-EEG_Conformer-blue?logo=github)](https://github.com/huiningjiao02-ship-it/EEG-Conformer-Replication-Cross-Dataset-Adaptation-for-Motor-Imagery-Decoding)
[![GitHub Repo](https://img.shields.io/badge/GitHub-EEG_Emotion_LibEER-green?logo=github)](https://github.com/huiningjiao02-ship-it/EEG-based-Emotion-Recognition-Aligned-with-IEEE-TAFFC-2025-LibEER-Benchmark-Introduction)

This repository contains two independent EEG deep learning projects:

1. **EEG-Conformer** – Motor imagery decoding on PhysioNet eegmmidb  
2. **EEG Emotion Recognition** – Three‑class emotion classification aligned with the LibEER benchmark

---

## 🧠 Project 1: EEG‑Conformer for Motor Imagery

[![View Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/huiningjiao02-ship-it/EEG-Conformer-Replication-Cross-Dataset-Adaptation-for-Motor-Imagery-Decoding)

### Introduction

This work reproduces and fully adapts the **EEG‑Conformer** architecture (Song et al., 2023) on the **PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb)**. The original model was extended from 22 to **64 channels** to match the dataset, and the hybrid CNN‑Transformer architecture is evaluated for motor imagery decoding (four‑class: left/right/both fists, both feet).

The project verifies the classification performance and generalization capability of this architecture on a widely used open‑source EEG dataset.

### 项目介绍

本工作在 **PhysioNet EEG 运动想象数据集（eegmmidb）** 上完整复现并适配了 **EEG‑Conformer** 模型。原模型从 22 通道扩展至 **64 通道**，并对该 CNN‑Transformer 混合架构在通用开源 EEG 数据上的分类性能与泛化能力进行了验证（四分类：左手 / 右手 / 双手 / 双脚）。

---

## 😊 Project 2: EEG‑based Emotion Recognition (LibEER Benchmark)

[![View Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/huiningjiao02-ship-it/EEG-based-Emotion-Recognition-Aligned-with-IEEE-TAFFC-2025-LibEER-Benchmark-Introduction)

### Introduction

This project builds a lightweight **1D‑CNN** classification pipeline using a public emotional EEG dataset (Kaggle: *Feeling Emotions*). It performs end‑to‑end offline three‑class emotion recognition (Negative / Neutral / Positive) and aligns its evaluation protocol with the standardized framework proposed in the **LibEER** benchmark (IEEE TAFFC 2025).

The goal is to provide a **rapidly reproducible and easily extensible** baseline model with a clean, engineered code framework.

### 项目介绍

本项目基于公开情绪 EEG 数据集（Kaggle: *Feeling Emotions*）搭建了一个轻量级 **1D‑CNN** 分类流程，实现端到端的离线三分类情绪识别（消极 / 中性 / 积极）。评估协议与 **LibEER** 基准框架（IEEE TAFFC 2025）保持一致。

项目旨在提供一个**可快速复现、易于扩展**的基线模型，并附带工程化的代码框架。

---

## 📁 Repository Overview

| Project | Description | GitHub Link |
| :--- | :--- | :--- |
| EEG‑Conformer | Motor imagery decoding on eegmmidb (64‑ch) | [Repo](https://github.com/huiningjiao02-ship-it/EEG-Conformer-Replication-Cross-Dataset-Adaptation-for-Motor-Imagery-Decoding) |
| EEG Emotion (LibEER) | 3‑class emotion classification with 1D‑CNN | [Repo](https://github.com/huiningjiao02-ship-it/EEG-based-Emotion-Recognition-Aligned-with-IEEE-TAFFC-2025-LibEER-Benchmark-Introduction) |

---

## 🚀 Quick Links

- **EEG‑Conformer Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/1_5OjmRb-AIt7KSR5zOXs8lNSYDNrqqce)
- **Emotion Recognition Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/1GnnSo4iL_0Os2eptGpwOEsxCbBS6Vq6V)

---

## 📄 License

Both projects are open‑source under the MIT License. See individual repositories for details.
