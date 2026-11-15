# Machine Learning Classifier Comparison

## Project Overview

This project implements and compares three machine learning classification algorithms: **Random Forest (RF), Gaussian Mixture Models (GMM), and Multilayer Perceptron (MLP)**.

The objective is to evaluate their classification performance across different datasets, examine the effects of hyperparameter tuning, and compare their accuracy, computational efficiency, and generalization performance.

The experiments use two benchmark datasets from the UCI Machine Learning Repository.

## Datasets

**1. Optical Recognition of Handwritten Digits (Optdigits)**
- 5,620 samples
- 64 numerical features
- Handwritten digit classification

**2. Statlog (Landsat Satellite)**
- 6,435 samples
- 36 numerical features
- Satellite image classification

Together, the datasets contain **12,055 samples**.

## Machine Learning Models

### 1. Random Forest
- Implemented an ensemble-based classification model.
- Performed randomized hyperparameter search to optimize tree count, maximum depth, feature selection, and minimum leaf size.
- Analyzed feature importance and out-of-bag (OOB) error.

### 2. Gaussian Mixture Model (GMM)
- Implemented a generative classifier using class-specific Gaussian mixture models.
- Experimented with different numbers of Gaussian components, covariance types, and initialization strategies.
- Analyzed log-likelihood convergence and model selection using AIC and BIC.

### 3. Multilayer Perceptron (MLP)
- Implemented a neural network classifier trained using backpropagation.
- Experimented with hidden-layer architectures, activation functions, learning rates, and regularization.
- Analyzed training and validation loss curves to investigate convergence and overfitting.

## Methodology

1. **Data Exploration:** Examined dataset characteristics, feature distributions, and class balance.
2. **Preprocessing:** Checked for missing values, standardized numerical features, and encoded class labels.
3. **Model Training:** Trained Random Forest, GMM, and MLP classifiers.
4. **Hyperparameter Tuning:** Explored model configurations to improve classification accuracy.
5. **Cross-Validation:** Used stratified 5-fold cross-validation to compare performance across data splits.
6. **Evaluation:** Compared accuracy, recall, specificity, F1-score, AUC-ROC, training time, and inference time.

## Experimental Results

The following results were reported by the notebook's cross-validation experiments.

| Model | Optdigits Accuracy | Statlog Accuracy |
|---|---|---|
| Random Forest | 98.36% | 92.00% |
| GMM | 96.01% | 88.13% |
| MLP | 98.13% | 90.49% |

**Key Findings**
- Random Forest achieved the highest reported accuracy on both datasets.
- MLP delivered competitive classification performance, particularly on Optdigits.
- GMM provided an alternative probabilistic classification approach, although its reported accuracy was lower.
- Training and inference times varied across models and datasets.

**Evaluation limitation:** The current notebook standardizes features before cross-validation and uses overlapping folds for hyperparameter selection and reported evaluation. These choices may lead to optimistic performance estimates. The results should be revalidated using fold-specific preprocessing and a separate evaluation procedure before being treated as unbiased generalization estimates.

## Technologies Used

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Datasets:** UCI Machine Learning Repository (`ucimlrepo`)
- **Development Environment:** Jupyter Notebook

## How to Run

1. Clone or download this repository.
2. Open `ML_Classification_Comparison.ipynb` in Jupyter Notebook or Google Colab.
3. Install the required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `ucimlrepo`.
4. Execute the notebook cells sequentially.

The notebook fetches both datasets through `ucimlrepo`. An internet connection is required for initial dataset retrieval.

## Project Outputs

The notebook includes:
- Dataset and class-distribution visualizations
- Hyperparameter optimization results
- Random Forest feature-importance and OOB analyses
- GMM convergence and model-selection analyses
- MLP training and validation loss visualizations
- ROC curves comparing the three classifiers
- Comparative performance tables

The evaluation code also generates `final_evaluation_raw_metrics.csv` and `final_comparison_table.csv`.

## Author

**Omar Hesham**

Data Science Engineering Student — German International University (GIU)
