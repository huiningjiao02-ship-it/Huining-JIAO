# Huining-JIAO

- **Project**: Binary Classification of Motor Imagery from EEG Signals
- **Skills & Methods**: Proficient in MNE for building complete EEG analysis workflows (data acquisition → preprocessing → feature extraction → model training → visualization). Mastered classical EEG algorithms including CSP for spatial filtering and LDA for classification.
- **Core Work**:
  1. Efficiently interfaced with public EEG datasets (e.g., BCI Competition IV 2a).
  2. Implemented high-quality preprocessing pipelines (filtering, rereferencing, epoching).
  3. Applied EEG decoding: optimized spatial feature extraction using CSP (via grid search) combined with LDA classifiers.
  4. Conducted comprehensive evaluation (within-subject & cross-subject analysis, cross-validation) and visualization (topomaps, time-frequency analysis, parameter tuning curves).
- **Key Achievements**:
  1. Established a valid end-to-end pipeline, achieving classification accuracy significantly above chance (~67%) in within-subject analysis.
  2. Identified and analyzed the key challenge of model performance degradation in naive cross-subject data merging (~44% accuracy), providing clear direction for future improvement using feature alignment or domain adaptation techniques.
  3. Created a reusable, modular EEG analysis template that ensures reproducibility.

**Project Title:** Heart Disease Prediction – Binary Classification Project  

**Skills Applied:**  
- Utilized Python data science stack (pandas, numpy, scikit-learn) for data loading, cleaning, and standardization  
- Manually implemented a Simple Perceptron from scratch to understand the classical machine learning update mechanism  
- Leveraged scikit-learn's Perceptron and Logistic Regression for model comparison and performance benchmarking  
- Evaluated models using accuracy, recall, F1-score, and confusion matrices  
- Incorporated the concept of “data quality control” from medical testing into the preprocessing pipeline  

**Core Work Content:**  
1. **Dataset Preprocessing:**  
   - Loaded UCI Heart Disease dataset (303 samples, 13 features) via `ucimlrepo` for reliable metadata handling  
   - Handled missing values by removing incomplete rows  
   - Converted original multi-class target (0–4) into binary labels (0 = healthy, 1 = heart disease) to reflect real diagnostic scenarios  
   - Standardized all features to zero mean and unit variance for stable gradient-based training  

2. **Model Development & Comparison:**  
   - **Simple Perceptron (from scratch):** Implemented using classical update rule; labels transformed to {-1, +1}; trained for 20 epochs with misclassification tracking  
   - **Scikit-learn Perceptron:** Employed SGD optimizer for efficient training  
   - **Logistic Regression:** Used as a robust probabilistic linear baseline  
   - Applied stratified train-test split (`random_state` fixed) to maintain class distribution consistency  

3. **Model Evaluation & Selection:**  
   - Test accuracy: Simple Perceptron (70.00%), sklearn Perceptron (78.33%), Logistic Regression (83.33%)  
   - Logistic Regression showed strong recall: 28/32 for healthy, 22/28 for diseased, with low false negatives (6/28)  
   - Selected Logistic Regression as the final model due to its balanced precision-recall trade-off and interpretability  

**Achievement Value:**  
- Achieved **high test accuracy of 83.33%** (50/60), demonstrating potential for clinical decision support  
- Clearly illustrated performance evolution from classical to modern linear models, highlighting logistic regression’s reliability in medical binary classification  
- Embedded “data quality control” principles throughout the pipeline, enhancing real-world applicability and trustworthiness  
- Provided an extensible baseline framework for future work, such as incorporating nonlinear models, cross-validation, and larger datasets  
assessment with clarity, reproducibility, and diagnostic relevance.

Based on the provided training logs, evaluation results, and project description, here is a detailed and structured elaboration of the project specifics in English, highlighting technical depth, quantitative outcomes, and process innovation.

---

### **Project Title**: Development and Process Automation of an Image Binary Classification System Based on Deep Learning

### **II. Core Technology Stack**
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch, TorchVision
- **Model Architecture**: Transfer learning model based on AlexNet
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC
- **Environment**: Linux / Python 3.12 / CUDA (if applicable)

---

### **III. Core Work & Innovations**

#### 1. **Automated Data Preprocessing Pipeline**
- Developed an automated data preprocessing pipeline supporting image augmentation, normalization, and dataset splitting (train/val/test).
- Implemented a **one-click data preparation** system, reducing manual intervention time from **4 hours per run to 30 minutes per run**, significantly improving experimental iteration efficiency.

#### 2. **Model Development & Training Optimization**
- Adapted AlexNet via transfer learning for a binary classification task (e.g., a prototype "cat vs. dog" task, analogous to "diseased vs. normal" in medical imaging).
- Designed and implemented training strategies including **dynamic learning rate adjustment, early stopping, and loss function monitoring**.
- Monitored multiple metrics throughout training (Loss, Accuracy, ROC-AUC, PR-AUC, F1, etc.) to ensure stable model convergence.

#### 3. **End-to-End Training Pipeline Design**
- Implemented automated training logging (outputting loss, accuracy, timestamps, etc., per epoch).
- Incorporated features for model checkpoint resumption and best-model saving (e.g., `epoch6.pt`).
- Achieved **100% accuracy on both training and validation sets by epoch 7**, demonstrating rapid model convergence.

#### 4. **Model Evaluation & Finalization**
- Validated model performance on an independent test set, achieving the following **perfect evaluation results**:
  - **Test Accuracy: 100%**
  - **Precision: 1.0000**
  - **Recall: 1.0000**
  - **F1 Score: 1.0000**
  - **ROC-AUC: 1.0000**
- Generated a detailed classification report supporting per-class metric analysis (e.g., balanced performance for both "cat" and "dog" classes).

#### 5. **System Fault Tolerance & User Guidance**
- Integrated exception handling mechanisms (e.g., logging keyboard interrupts).
- Provided clear command-line output and structured logs for easy user monitoring and debugging.


### **V. Project Value & Extended Significance**
- **Successful Technical Validation**: Demonstrated the high efficacy of transfer learning for **small-sample image classification**, establishing a technical base for future medical imaging AI models (e.g., histopathology slide classification, X-ray recognition).
- **Process Engineering**: Achieved full automation of the "data → model → evaluation" pipeline, ensuring high reproducibility and scalability.
- **Integration of Academia & Engineering**: Showcased both the powerful performance of deep learning models and the emphasis on efficiency, stability, and user experience in engineering implementation.
