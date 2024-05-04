# Medical Insurance Cost Prediction using Machine Learning

## Project Overview
The "Medical Insurance Cost Prediction using Machine Learning" project is designed to predict individual medical insurance costs based on various demographic and health-related factors. Using data-driven insights, this project aims to assist healthcare providers, insurance companies, and individuals in understanding and forecasting medical expenses.

## Dataset
The dataset used in this project consists of 2,772 entries with 7 features:
- `age`: Age of the primary beneficiary.
- `sex`: Gender of the primary beneficiary.
- `bmi`: Body Mass Index, which provides a sense of an individual's body weight adjusted for height.
- `children`: Number of children/dependents covered by the insurance plan.
- `smoker`: Smoking status of the beneficiary.
- `region`: The beneficiary’s residential area in the US (northeast, southeast, southwest, northwest).
- `charges`: Individual medical costs billed by health insurance.

## Introduction

Healthcare costs are a significant concern for individuals and families worldwide. Predicting medical insurance costs accurately can help insurance companies determine premiums and assist individuals in planning their healthcare expenses. This project focuses on building machine learning models to predict insurance costs based on demographic and health-related attributes.

## Problem Statement

1. What are the most important factors that affect medical expenses?
2. How well can machine learning models predict medical expenses?
3. How can machine learning models be used to improve the efficiency and profitability of health insurance companies?

## Features

- **Data Exploration**: Explore the dataset to understand its structure, identify missing values, and analyze the distribution of features.
- **Data Preprocessing**: Prepare the data by handling categorical variables, renaming columns, and scaling numerical features.
- **Model Training**: Utilize various machine learning models to train predictive models on the prepared dataset.
- **Model Evaluation**: Evaluate model performance using metrics such as R-squared score and mean squared error to assess predictive accuracy
  
## Technologies Used
- Python 3.8+
- Pandas for data manipulation
- Matplotlib and Seaborn for data visualization
- Scikit-learn for implementing machine learning models

## Installation
To run this project, you will need Python installed on your system. Clone the repository and install the required packages:
```bash
git clone https://github.com/rahulvyasm/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
pip install -r requirements.txt
```

## Usage
To use the scripts in this project:
1. Navigate to the cloned directory.
2. Run the Jupyter Notebook for an interactive session:
   ```bash
   jupyter notebook 95-medical-insurance-price-prediction.ipynb
   ```
3. To execute a Python script directly:
   ```bash
   python medical_insurance_price_predictor.py
   ```

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Ensure to update tests as appropriate.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact Information
For any queries or assistance, please contact [me@rahulvyasm.com](mailto:me@rahulvyasm.com).
