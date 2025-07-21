# Predicting Baseline MMSE Scores using Clinical and Imaging Biomarkers from the ADNI Dataset

## Introduction:
This project aims to develop machine learning regression model to predict baseline Mini-Mental State Examination (MMSE) scores using a combination of demographic, clinical, and neuroimaging biomarkers available in the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. MMSE is a widely used screening tool to measure cognitive impairment. Early and accurate estimation of MMSE can aid in timely clinical intervention and support diagnostic decision-making in the context of Alzheimer’s Disease (AD).

## Dataset and Processing:  
The project uses the ADNI baseline dataset, focusing on the following categories:

1. **Demographic Feature**:
 - AGE

2. **Genetic Risk Factor**:
 - APOE4 (0, 1, 2 copies of the E4 allele)

3. **Clinical Cognitive Assessments**: 
 - ADAS11_bl (Alzheimer's Disease Assessment Scale)
 - ADAS13_bl (Expanded version)
 - CDRSB_bl (Clinical Dementia Rating – Sum of Boxes at baseline)
 - RAVLT_immediate_bl, RAVLT_learning_bl, RAVLT_forgetting_bl, RAVLT_perc_forgetting_bl(Rey Auditory Verbal Learning Test)
 - FAQ_bl (Functional Activities Questionnaire)

4. **Imaging Biomarkers**:  
 - Ventricles_bl, Hippocampus_bl, WholeBrain_bl, Entorhinal_bl, Fusiform_bl, MidTemp_bl

5. **Target Variable**:
 - MMSE_bl (Mini-Mental State Examination score at baseline)


Data was preprocessed for missing values, encoded appropriately, and scaled. MongoDB was used as the document-oriented database to store structured data records extracted from CSV files.
### Preprocessing Pipelines:
- Categorical Pipeline (APOE4): Imputed using most frequent values and encoded using OneHotEncoder.
- Clinical Pipeline: Median imputation and StandardScaler.
- Imaging Pipeline: KNNImputer (k=5) followed by StandardScaler.
- Age Pipeline: Only scaled.


## Model Experiments and Performance Summary:  
Seven regression models were evaluated:
- Linear Regression (Default)
- Ridge Regression (Alpha + Max Iter tuned)
- Lasso Regression (Alpha + Max Iter tuned)
- Support Vector Regressor (Kernel + C + Gamma tuned)
- Random Forest (Depth + Estimators tuned)
- Gradient Boosting (Learning Rate + Depth + Estimators tuned)
- XGBoost (Learning Rate + Depth + Estimators tuned)

GridSearchCV was applied to all models to find the best parameters. Performance metrics (R2, RMSE, MAE) were used to evaluate for test sets.
<table>
  <tr>
    <th>Model</th>
    <th>R2</th>
    <th>RMSE</th>
    <th>RMAE</th>
  </tr>
  <tr>
    <td>Ridge Regression</td>
    <td>0.670</td>
    <td>1.565</td>
    <td>1.224</td>
  </tr>
  <tr>
    <td>Lasso Regression </td>
    <td>0.669</td>
    <td>1.567</td>
    <td>1.224</td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>0.668</td>
    <td>1.568</td>
    <td>1.225</td>
  </tr>
  <tr>
    <td>Support Vector Regressor</td>
    <td>0.665</td>
    <td>1.577</td>
    <td>1.226</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.660</td>
    <td>1.589</td>
    <td>1.253</td>
  </tr>
  <tr>
    <td>Gradient Boosting</td>
    <td>0.656</td>
    <td>1.598</td>
    <td>1.256</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>0.655</td>
    <td>1.599</td>
    <td>1.264</td>
  </tr>
</table>


## Machine Learning Lifecycle:
- Data Ingestion 
- Data validation
- Data transformation
- Model trainer
- Model pusher

## Code Deployment:
https://mmse-prediction.onrender.com

