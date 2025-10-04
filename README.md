# Machine-Learning-Pipeline-for-Titanic-Survival-Prediction
This project implements a complete machine learning pipeline for predicting passenger survival on the Titanic using scikit-learn. The pipeline includes data preprocessing, feature engineering, model training, and evaluation.
Machine Learning Pipeline for Titanic Survival Prediction
Overview
This project implements a complete machine learning pipeline for predicting passenger survival on the Titanic using scikit-learn. The pipeline includes data preprocessing, feature engineering, model training, and evaluation.

Dataset
The project uses the classic Titanic dataset (train.csv) containing information about Titanic passengers including:

PassengerId: Unique identifier

Survived: Target variable (0 = No, 1 = Yes)

Pclass: Ticket class (1st, 2nd, 3rd)

Name: Passenger name

Sex: Gender

Age: Age in years

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Ticket: Ticket number

Fare: Passenger fare

Cabin: Cabin number

Embarked: Port of embarkation

Data Preprocessing
Features Removed
PassengerId, Name, Ticket, Cabin - removed as they don't contribute to predictive power

Pipeline Components
Imputation Transformer (trf1):

Imputes missing Age values

Imputes missing Embarked values with most frequent category

One-Hot Encoding (trf2):

Encodes categorical variables Sex and Embarked

Feature Scaling (trf3):

Applies MinMaxScaler to normalize features

Feature Selection (trf4):

Selects top 8 features using chi-squared test

Model Training (trf5):

Decision Tree Classifier

Model Performance
Test Accuracy: 62.57%

Cross-Validation Score: 63.91%

Best Parameters from GridSearch: max_depth=2

Key Features
Pipeline Implementation
The project demonstrates two ways to create pipelines:

Pipeline() with named steps

make_pipeline() for simplified syntax

Model Evaluation
Cross-validation with 5 folds

GridSearchCV for hyperparameter tuning

Accuracy scoring for model performance

Export and Deployment
Model exported as pipe.pkl using pickle

Example inference shown with test input

Usage
Loading the Model
python
import pickle
pipe = pickle.load(open('pipe.pkl', 'rb'))
Making Predictions
python
import numpy as np
test_input = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)
prediction = pipe.predict(test_input)
Dependencies
numpy

pandas

scikit-learn

pickle

Files
machine_learning_last.ipynb: Main notebook with complete implementation

pipe.pkl: Serialized pipeline model

train.csv: Titanic dataset

Results
The optimized Decision Tree model with max_depth=2 achieved approximately 64% accuracy through cross-validation, providing a reasonable baseline for Titanic survival prediction while avoiding overfitting.

This project serves as a comprehensive example of building end-to-end machine learning pipelines with proper preprocessing, model selection, and deployment capabilities.
