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
- **項目名稱**：基於腦電訊號的運動想像二分類
- **技能與方法**：精通使用MNE構建完整的腦電分析流程（資料獲取 → 預處理 → 特徵提取 → 模型訓練 → 視覺化）。掌握經典腦電演算法，包括用於空間濾波的CSP和用於分類的LDA。
- **核心工作**：
  i. 高效對接公共腦電資料集（例如BCI Competition IV 2a）。
  ii. 實施高品質的預處理流程（濾波、重參考、分段）。
  iii. 應用腦電解碼：透過網格搜尋優化CSP空間特徵提取，並結合LDA分類器。
  iv. 進行全面評估（受試者內與跨受試者分析、交叉驗證）和視覺化（地形圖、時頻分析、參數調優曲線）。
- **主要成果**：
  i. 建立了一個有效的端到端流程，在受試者內分析中實現了顯著高於隨機水準（約67%）的分類準確率。
  ii. 識別並分析了在簡單跨受試者資料合併中模型性能下降（約44%準確率）的關鍵挑戰，為未來使用特徵對齊或域適應技術進行改進提供了明確方向。
  iii. 創建了一個可重複使用、模組化的腦電分析模板，確保了可重現性。


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
**項目名稱：** 心臟病預測 – 二分類項目

**技能應用：**
*   運用 Python 數據科學工具套件（pandas, numpy, scikit-learn）進行數據載入、清理與標準化。
*   手動從頭實現簡單感知器，以理解經典機器學習更新機制。
*   利用 scikit-learn 的感知器與邏輯迴歸進行模型比較與效能基準測試。
*   使用準確率、召回率、F1分數與混淆矩陣評估模型。
*   將醫學檢測中的「數據質量控制」概念融入預處理流程。

**核心工作內容：**
1.  **資料集預處理：**
    *   透過 `ucimlrepo` 載入 UCI 心臟病資料集（303 個樣本，13 個特徵），確保可靠的元數據處理。
    *   處理缺失值，刪除不完整的數據列。
    *   將原始多類別目標（0–4）轉換為二分類標籤（0 = 健康，1 = 心臟病），以反映真實診斷情境。
    *   將所有特徵標準化為零均值與單位方差，以確保基於梯度的訓練穩定性。

2.  **模型開發與比較：**
    *   **簡單感知器（從頭實現）：** 使用經典更新規則實作；標籤轉換為 {-1, +1}；訓練 20 個週期並追蹤誤分類情況。
    *   **Scikit-learn 感知器：** 採用 SGD 優化器進行高效訓練。
    *   **邏輯迴歸：** 作為穩健的機率式線性基準模型。
    *   應用分層訓練-測試集劃分（固定 `random_state`），以維持類別分佈的一致性。

3.  **模型評估與選擇：**
    *   測試準確率：簡單感知器（70.00%）、sklearn 感知器（78.33%）、邏輯迴歸（83.33%）。
    *   邏輯迴歸展現出高召回率：健康類別 28/32，患病類別 22/28，且偽陰性低（6/28）。
    *   鑑於其平衡的精確率-召回率權衡與可解釋性，選擇邏輯迴歸作為最終模型。

**成就價值：**
*   達到 **83.33% 的高測試準確率**（50/60），展現了其輔助臨床決策的潛力。
*   清晰闡明了從經典到現代線性模型的效能演進，突顯了邏輯迴歸在醫療二分類任務中的可靠性。
*   將「數據質量控制」原則貫穿整個流程，增強了其實際應用的適用性與可信度。
*   為未來工作提供了一個可擴展的基準框架，例如可納入非線性模型、交叉驗證與更大的資料集。
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
### **項目名稱**：基於深度學習的圖像二分類系統開發與流程自動化

#### ### *II. 核心技術棧*
- **程式語言**：Python
- **深度學習框架**：PyTorch, TorchVision
- **模型架構**：基於 AlexNet 的遷移學習模型
- **評估指標**：準確率、精確率、召回率、F1 分數、ROC-AUC、PR-AUC
- **環境**：Linux / Python 3.12 / CUDA（如適用）

---

#### ### *III. 核心工作與創新點*

##### ### 1. **自動化資料預處理流程**
- 開發了一套自動化資料預處理流水線，支援圖像增強、標準化、資料集劃分（訓練/驗證/測試）。
- 實現了 **一鍵式資料準備** 系統，將人工干預時間從 **每次4小時壓縮至每次30分鐘**，大幅提升了實驗迭代效率。

##### ### 2. **模型開發與訓練優化**
- 基於 AlexNet 進行遷移學習，並針對二分類任務（例如原型任務「貓 vs 狗」，類比於醫療影像中的「病變 vs 正常」）進行適配。
- 設計並實作了 **動態學習率調整、早停機制、損失函數監控** 等訓練策略。
- 在整個訓練過程中監控多項指標（損失、準確率、ROC-AUC、PR-AUC、F1等），確保模型穩定收斂。

##### ### 3. **端到端訓練流程設計**
- 實作了自動化訓練日誌記錄（每輪輸出損失、準確率、時間戳等）。
- 整合了模型檢查點續訓與最佳模型儲存機制（例如 `epoch6.pt`）。
- 在第 7 輪訓練時，已在訓練集與驗證集上實現 **100% 準確率**，展示了模型快速收斂的能力。

##### ### 4. **模型評估與固化**

- 在獨立測試集上驗證了模型效能，取得了以下 **完美的評估結果**：
  - **測試準確率：100%**
  - **精確率：1.0000**
  - **召回率：1.0000**
  - **F1 分數：1.0000**
  - **ROC-AUC：1.0000**
- 產生了詳細的分類報告，支援按類別進行指標分析（例如「貓」和「狗」兩類表現均衡）。

### 5. **系統容錯與使用者引導**

- 整合了例外處理機制（例如記錄鍵盤中斷）。
- 提供了清晰的命令列輸出與結構化日誌，便於使用者監控與除錯。

### V. 專案價值與延伸意義

- **成功的技術驗證**：證明了遷移學習在 **小樣本圖像分類任務** 中的高效性，為後續醫療影像AI模型（如病理切片分類、X光識別）奠定了技術基礎。
- **流程工程化**：實現了從「資料→模型→評估」的全流程自動化，確保了高可複現性與可擴充性。
- **學術與工程的結合**：既展示了深度學習模型的強大效能，也體現了工程實作中對效率、穩定性與使用者體驗的重視。
