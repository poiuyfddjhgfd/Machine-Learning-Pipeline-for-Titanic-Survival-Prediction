## Titanic Survival Prediction - Machine Learning Pipeline  
#  Project Overview
A complete machine learning pipeline for predicting passenger survival on the Titanic using scikit-learn. This project demonstrates data preprocessing, feature engineering, model training, and deployment.
# Installation
pip install numpy pandas scikit-learn
# Basic Usage
import pickle
import numpy as np

# Load the trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Make a prediction
test_input = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)
prediction = pipe.predict(test_input)
print(f"Survival prediction: {prediction[0]}")  # 0 = Did not survive, 1 = Survived
# 📊 Dataset Features
📊 Dataset Features
Features Used:

Pclass: Ticket class (1st, 2nd, 3rd)

Sex: Gender

Age: Age in years

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Fare: Passenger fare

Embarked: Port of embarkation

# Target Variable:

Survived: 0 = No, 1 = Yes
# 🔧 Pipeline Architecture
The machine learning pipeline consists of 5 sequential steps:
1. Data Imputation (trf1)
ColumnTransformer([
    ('impute_age', SimpleImputer(), [2]),           # Impute missing Age
    ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])  # Impute Embarked
], remainder='passthrough')
2. One-Hot Encoding (trf2)
3. ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(
        sparse_output=False, 
        handle_unknown='ignore'
    ), [1, 6])
], remainder='passthrough')
3. Feature Scaling (trf3)
4. ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0, 10))

])4. Feature Selection (trf4)
SelectKBest(score_func=chi2, k=8)
5. Model Training (trf5)
DecisionTreeClassifier()
📈 Model Performance
Metric	Score
Test Accuracy	62.57%
Cross-Validation Score	63.91%
Best Parameters	max_depth=2
🎯 Hyperparameter Tuning
# GridSearchCV for optimal parameters
params = {
    'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, None]
}

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)

print(f"Best score: {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# 🔍 Model Inspection
Access Pipeline Components
# Get all pipeline steps
steps = pipe.named_steps

# Access specific transformer statistics
embarked_stats = pipe.named_steps['columntransformer-1'].transformers_[1][1].statistics_
print(f"Most frequent Embarked value: {embarked_stats}")
# Visualize Pipeline
from sklearn import set_config
set_config(display='diagram')
pipe  # Display interactive pipeline diagram
 # Export/Import Model
 Export Pipeline
import pickle
pickle.dump(pipe, open('pipe.pkl', 'wb'))
Load Pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))
# Test case 1: Female, 1st class, high fare
test_input1 = np.array([1, 'female', 28.0, 0, 0, 100.0, 'C'], dtype=object).reshape(1, 7)
pred1 = pipe.predict(test_input1)  # Likely survives (1)

# Test case 2: Male, 3rd class, low fare  
test_input2 = np.array([3, 'male', 35.0, 0, 0, 7.25, 'S'], dtype=object).reshape(1, 7)
pred2 = pipe.predict(test_input2)  # Likely doesn't survive (0)
 # Development
Training the Model
# Load and prepare data
df = pd.read_csv('train.csv')
df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)

# Split features and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train pipeline
pipe = make_pipeline(trf1, trf2, trf3, trf4, trf5)
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# 🎯 Key Insights
Feature Importance: Sex, Pclass, and Fare are most predictive of survival

Model Simplicity: Decision Tree with max_depth=2 prevents overfitting

Pipeline Efficiency: All preprocessing steps are encapsulated for easy deployment
# 🤝 Contributing
Feel free to:

Experiment with different models

Add feature engineering techniques

Improve hyperparameter tuning

Enhance documentation

