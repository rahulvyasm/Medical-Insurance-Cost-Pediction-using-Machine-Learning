#Importing Neccessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, roc_curve, auc

df = pd.read_csv("medical_insurance.csv")

# Preprocessing
categorical_features = ['sex', 'smoker', 'region']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
X_transformed = transformer.fit_transform(df.drop('charges', axis=1))
y = df['charges']

# Split the dataset into training and testing sets for regression
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Models for regression
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
}

# Train and evaluate regression models
print("Regression Models Evaluation:")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2_score_model = r2_score(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
    mae_score = mean_absolute_error(y_test, predictions)

    print("Model:", name)
    print("R2 Score:", r2_score_model)
    print("MSE Score:", mse_score)
    print("MAE Score:", mae_score)
    print()

# Preparing for classification
df['charges_category'] = (df['charges'] > df['charges'].median()).astype(int)
y_categorical = df['charges_category']
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_transformed, y_categorical, test_size=0.2, random_state=42)

# Models for classification
class_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classification": KNeighborsClassifier(),
    "SVM Classification": SVC(probability=True),
    "Naive Bayes Classification": GaussianNB()
}

# Train and evaluate classification models
print("\nClassification Models Evaluation:")
for name, model in class_models.items():
    model.fit(X_train_cat, y_train_cat)
    if name == "SVM Classification":
        predictions = model.decision_function(X_test_cat)
    else:
        predictions = model.predict_proba(X_test_cat)[:, 1]
    accuracy = accuracy_score(y_test_cat, model.predict(X_test_cat))
    print(f"{name}: Accuracy = {accuracy:.2f}")

# ROC Curve Analysis
plt.figure(figsize=(10, 8))
for name, model in class_models.items():
    if name == "SVM Classification":
        predictions = model.decision_function(X_test_cat)
    else:
        predictions = model.predict_proba(X_test_cat)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_cat, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()
print('\n')

# User Input Function
def get_user_input():
    age = int(input("Enter age: "))
    sex = input("Enter sex (male/female): ").lower()
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter number of children: "))
    smoker = input("Enter smoker (yes/no): ").lower()
    region = input("Enter region (southwest/southeast/northwest/northeast): ").lower()

    return age, sex, bmi, children, smoker, region

# Get user input
age, sex, bmi, children, smoker, region = get_user_input()

# Create a sample input (like a DataFrame)
user_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

# Apply the same transformations
user_data_transformed = transformer.transform(user_data)

# Regression Predictions
print("\nRegression Predictions:")
for name, model in models.items():
    prediction = model.predict(user_data_transformed)[0]  # Get the single prediction
    print(f"{name}: Predicted charges: ${prediction:.2f}")

# Classification Predictions
print("\nClassification Predictions:")
for name, model in class_models.items():
    prediction = model.predict(user_data_transformed)[0]
    if prediction == 1:
        pred_class = "High charges"
    else:
        pred_class = "Low charges"
    print(f"{name}: Predicted category: {pred_class}") 
