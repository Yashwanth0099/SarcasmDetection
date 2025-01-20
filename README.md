# Sarcasm Detection in Reddit Comments

## Project Overview
- Developed machine learning models to detect sarcasm in Reddit comments.
- Dataset: 1,010,826 entries with 10 features, sourced from Kaggle.
- Objective: Analyze and classify comments as sarcastic or non-sarcastic.

## Data Preparation
- Removed 55 entries with missing values (<1% of data).
- Conducted EDA:
  - Balanced dataset: ~50% sarcastic, ~50% non-sarcastic comments.
  - Correlation analysis: High correlation (0.85) between "ups" and "score."
  - Word and sentence length analysis showed similar patterns for sarcastic and non-sarcastic comments.

## Algorithms and Tuning
1. **Grid Search**:
   - Logistic Regression and Decision Tree models were tuned for optimal hyperparameters.
2. **Random Search**:
   - Applied to Support Vector Classifier (SVC) for efficient hyperparameter exploration.
3. **Pipeline Implementation**:
   - TF-IDF Vectorizer combined with classifiers for modular and reproducible code.

## Models and Metrics
- **Logistic Regression**:
  - Accuracy: 69%, F1-Score: 0.68
- **Support Vector Machine (SVM)**:
  - Accuracy: 59%, F1-Score: 0.54
- **Decision Tree**:
  - Accuracy: 59%, F1-Score: 0.42
- **Bi-LSTM**:
  - Accuracy: 65%
- **Bi-GRU**:
  - Accuracy: 65%

## Explainability
- **SHAP Values**:
  - Used to interpret model predictions for Logistic Regression.
  - Visualized feature impact on predictions.

## Conclusion
- Logistic Regression outperformed other models with 69% accuracy.
- Neural models (Bi-LSTM, Bi-GRU) achieved competitive performance.
- Highlighted the importance of hyperparameter tuning and model explainability.
