#Importing Neccessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
df.head(5)

"""PRE-PROCESSING DATA"""

df.isnull().sum()

#get_dummies to strings, sex, smoker and region and concat with original df
df_dummied = pd.concat([df, pd.get_dummies(df[['sex', 'smoker', 'region']])], axis=1)

#drop the columns that i get_dummied
df_dummied.drop(columns=['sex', 'smoker', 'region'], inplace=True)

df_dummied.head()

"""# **EDA (EXPLORATORY DATA ANALYSIS)**"""

df.describe().T

#Subplots with hist
fig, axs = plt.subplots(5, figsize=(8,10))

axs[0].hist(df['age'], bins=20)
axs[0].set_title('Age Distribution')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Count')

axs[1].hist(df['children'], bins=20)
axs[1].set_title('Children Distribution')
axs[1].set_xlabel('Children')
axs[1].set_ylabel('Count')

axs[2].hist(df['smoker'], bins=20)
axs[2].set_title('Smoker Distribution')
axs[2].set_xlabel('Smoker')
axs[2].set_ylabel('Count')

axs[3].hist(df['sex'], bins=20)
axs[3].set_title('Sex Distribution')
axs[3].set_xlabel('Sex')
axs[3].set_ylabel('Count')

axs[4].hist(df['region'], bins=20)
axs[4].set_title('Region Distribution')
axs[4].set_xlabel('Region')
axs[4].set_ylabel('Count')

plt.tight_layout()

plt.show()

"""## **What are the primary factors influencing medical expenses?**"""

#heatmap
plt.figure(figsize=(12,9))
sns.heatmap(df_dummied.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

#scatter to smoker - charges
plt.scatter(df['smoker'], df['charges'], s=5)
plt.title("Smoker - Charges")
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.show()

#Charges mean between smokers and non smokers
smoker_charges = df[df["smoker"] == "yes"]["charges"]
non_smoker_charges = df[df["smoker"] == "no"]["charges"]

mean_smoker_charges = smoker_charges.mean()
mean_non_smoker_charges = non_smoker_charges.mean()

print(f"Mean charges for smokers: {mean_smoker_charges}")
print(f"Mean charges for non-smokers: {mean_non_smoker_charges}")

"""The principal factor to medical expenses are smokers, with 79% correlation, followed by age with 30% correlation and bmi with 20%.

## **More Statistics**
"""

#subplots bar
fig, axs = plt.subplots(3, figsize=(12,10))

axs[0].bar(df['age'], df['charges'])
axs[0].set_title('Age - Charges')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Charge')

axs[1].bar(df['age'], df['bmi'])
axs[1].set_title('Age - BMI')
axs[1].set_xlabel('Age')
axs[1].set_ylabel('BMI')

axs[2].bar(df['age'], df['children'])
axs[2].set_title('Age - Children')
axs[2].set_xlabel('Age')
axs[2].set_ylabel('Children')

plt.tight_layout()

plt.show()

#subplots bar
fig, axs = plt.subplots(2, figsize=(6,6))

axs[0].bar(df['sex'], df['bmi'])
axs[0].set_title('Sex - BMI')
axs[0].set_xlabel('Sex')
axs[0].set_ylabel('BMI')

axs[1].bar(df['smoker'], df['bmi'])
axs[1].set_title('Smoker - BMI')
axs[1].set_xlabel('Smoker')
axs[1].set_ylabel('BMI')

plt.tight_layout()

plt.show()

#female and male smokers
df.groupby(['smoker', 'sex']).size().unstack()

#subploits boxplots
fig, axs = plt.subplots(4, figsize=(5,8))

axs[0].boxplot(df['age'], vert=False)
axs[0].set_title('Age - BOXPLOT')

axs[1].boxplot(df['bmi'], vert=False)
axs[1].set_title('BMI - BOXPLOT')

axs[2].boxplot(df['children'], vert=False)
axs[2].set_title('Children - BOXPLOT')

axs[3].boxplot(df['charges'], vert=False)
axs[3].set_title('Charges - BOXPLOT')

plt.tight_layout()

plt.show()

"""The smokers make this Charges outliers, this will affect our model later, ill explain

# **Training and Testing Models**

## **How accurate are machine learning models in predicting medical expenses?**
"""

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
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()