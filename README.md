# Cause of Death Prediction - Machine Learning Project

## Overview

This project implements a comprehensive machine learning pipeline for predicting cause of death based on various demographic and health-related features. The analysis includes multiple classification algorithms with performance comparison and evaluation metrics.

## Project Structure

```
PR-Project/
├── PR_project.ipynb              # Main Jupyter notebook with analysis
├── Cause of Death_Training Part.csv  # Training dataset
├── pattern project documentation copy.docx  # Project documentation
├── pattern.pptx                  # Presentation slides
├── PR project.mp4               # Project video
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## Features

- **Data Preprocessing**: Comprehensive data cleaning, feature engineering, and outlier detection
- **Multiple ML Models**: Implementation of various classification algorithms:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Neural Network (MLP)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- **Model Evaluation**: Performance metrics including accuracy, confusion matrix, ROC curves, and classification reports
- **Feature Engineering**: One-hot encoding, standardization, and gender-based feature creation

## Dataset

The project uses a dataset containing various features (X1-X11) and a target variable for cause of death prediction. The dataset includes:
- Demographic information
- Health-related features
- Categorical and numerical variables

## Key Components

### Data Preprocessing
- Gender encoding (Male/Female to binary features)
- One-hot encoding for categorical variables
- Standardization of numerical features
- Outlier detection and removal using IQR method

### Model Training
- Train-test split for model validation
- Cross-validation for robust performance evaluation
- Hyperparameter tuning using GridSearchCV

### Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- ROC-AUC Score
- Classification Report
- Mean Squared Error (for regression tasks)

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and utilities
- **Jupyter Notebook**: Interactive development environment

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Project

1. Clone the repository:
```bash
git clone <repository-url>
cd PR-Project
```

2. Open the Jupyter notebook:
```bash
jupyter notebook PR_project.ipynb
```

3. Run all cells in the notebook to execute the complete analysis.

## Model Performance

The project compares multiple machine learning algorithms and provides detailed performance metrics for each model, helping to identify the best performing algorithm for the given dataset.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or contributions, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and research purposes. The models and predictions should not be used for medical diagnosis or treatment decisions. 