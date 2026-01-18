# Huining-JIAO

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
