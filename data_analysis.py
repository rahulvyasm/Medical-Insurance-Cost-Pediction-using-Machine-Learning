#Importing Neccessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("medical_insurance.csv")
print(df.head(5))
print('\n')

print(df.isnull().sum())
print('\n')

#get_dummies to strings, sex, smoker and region and concat with original df
df_dummied = pd.concat([df, pd.get_dummies(df[['sex', 'smoker', 'region']])], axis=1)

#drop the columns that i get_dummied
df_dummied.drop(columns=['sex', 'smoker', 'region'], inplace=True)

print(df_dummied.head())
print('\n')

print(df.describe().T)
print('\n')

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