# Task: AI/ML Internship Task 1 - Data Cleaning & Preprocessing
# Dataset: Titanic (titanic.csv)
# Objective: To clean and prepare the raw Titanic dataset for a machine learning model.

# Step 1: Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TITANIC DATA CLEANING & PREPROCESSING")
print("=" * 60)

try:
    # Load the dataset from data directory
    df = pd.read_csv('../data/titanic.csv')
    print("✓ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: 'titanic.csv' not found in the data directory.")
    print("Please ensure the Titanic dataset is available as '../data/titanic.csv'")
    exit()

# Initial exploration
print("\n" + "=" * 40)
print("STEP 1: INITIAL DATA EXPLORATION")
print("=" * 40)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values)
print(f"\nTotal missing values: {missing_values.sum()}")

# Step 2: Handle Missing Values
print("\n" + "=" * 40)
print("STEP 2: HANDLING MISSING VALUES")
print("=" * 40)

print("Before handling missing values:")
print(df.isnull().sum())

# Impute Age column with median
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)
print(f"✓ Age column imputed with median value: {age_median}")

# Impute Embarked column with mode
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)
print(f"✓ Embarked column imputed with mode value: {embarked_mode}")

# Drop Cabin column due to too many missing values
if 'Cabin' in df.columns:
    cabin_missing_pct = (df['Cabin'].isnull().sum() / len(df)) * 100
    print(f"Cabin column has {cabin_missing_pct:.1f}% missing values")
    df.drop('Cabin', axis=1, inplace=True)
    print("✓ Cabin column dropped")

print("\nAfter handling missing values:")
print(df.isnull().sum())

# Step 3: Convert Categorical Features to Numerical
print("\n" + "=" * 40)
print("STEP 3: CONVERTING CATEGORICAL TO NUMERICAL")
print("=" * 40)

print("Original categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
print(list(categorical_cols))

# Label Encoding for Sex column (binary categorical)
if 'Sex' in df.columns:
    label_encoder = LabelEncoder()
    df['Sex_encoded'] = label_encoder.fit_transform(df['Sex'])
    print(f"✓ Sex column label encoded: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# One-Hot Encoding for Embarked column
if 'Embarked' in df.columns:
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    print(f"✓ Embarked column one-hot encoded into: {list(embarked_dummies.columns)}")

# Drop original categorical columns after encoding
columns_to_drop = ['Sex', 'Embarked']
if 'Name' in df.columns:
    columns_to_drop.append('Name')
if 'Ticket' in df.columns:
    columns_to_drop.append('Ticket')

for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
        print(f"✓ Dropped original column: {col}")

# Step 4: Normalize/Standardize Numerical Features
print("\n" + "=" * 40)
print("STEP 4: STANDARDIZING NUMERICAL FEATURES")
print("=" * 40)

# Identify numerical columns for scaling
numerical_cols = ['Age', 'Fare']
existing_numerical_cols = [col for col in numerical_cols if col in df.columns]

if existing_numerical_cols:
    print(f"Standardizing columns: {existing_numerical_cols}")
    
    # Handle any remaining missing values in Fare
    if 'Fare' in df.columns and df['Fare'].isnull().sum() > 0:
        fare_median = df['Fare'].median()
        df['Fare'].fillna(fare_median, inplace=True)
        print(f"✓ Fare column imputed with median: {fare_median}")
    
    scaler = StandardScaler()
    df[existing_numerical_cols] = scaler.fit_transform(df[existing_numerical_cols])
    print(f"✓ Numerical features standardized using StandardScaler")
    
    # Show statistics after scaling
    print("\nScaled features statistics:")
    print(df[existing_numerical_cols].describe())

# Step 5: Visualize and Remove Outliers
print("\n" + "=" * 40)
print("STEP 5: VISUALIZING AND REMOVING OUTLIERS")
print("=" * 40)

# Create a copy for outlier removal
df_no_outliers = df.copy()

def remove_outliers_iqr(data, column):
    """Remove outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = len(data[(data[column] < lower_bound) | (data[column] > upper_bound)])
    
    # Remove outliers
    data_cleaned = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return data_cleaned, outliers_count, (lower_bound, upper_bound)

# Create visualizations and remove outliers
plt.figure(figsize=(15, 5))

outlier_cols = [col for col in existing_numerical_cols if col in df.columns]
total_outliers_removed = 0

for i, col in enumerate(outlier_cols, 1):
    # Create subplot for boxplot
    plt.subplot(1, len(outlier_cols), i)
    
    # Before outlier removal
    plt.boxplot([df[col]], labels=[f'{col}\n(Before)'])
    plt.title(f'Outliers in {col}')
    plt.ylabel('Standardized Values')
    
    # Remove outliers
    df_no_outliers, outliers_count, bounds = remove_outliers_iqr(df_no_outliers, col)
    total_outliers_removed += outliers_count
    
    print(f"✓ {col}: Removed {outliers_count} outliers (bounds: {bounds[0]:.2f} to {bounds[1]:.2f})")

plt.tight_layout()
plt.savefig('outlier_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ Outlier visualizations saved as 'outlier_visualization.png'")

# Create final cleaned dataset
df_cleaned = df_no_outliers.copy()

print(f"\nTotal outliers removed: {total_outliers_removed}")
print(f"Dataset shape after outlier removal: {df_cleaned.shape}")

# Step 6: Final Check
print("\n" + "=" * 40)
print("STEP 6: FINAL CHECK - CLEANED DATASET")
print("=" * 40)

print("Final cleaned dataset info:")
print(df_cleaned.info())

print("\nFinal dataset statistics:")
print(df_cleaned.describe())

print("\nMissing values in cleaned dataset:")
print(df_cleaned.isnull().sum())

print(f"\nFinal dataset shape: {df_cleaned.shape}")
print("Columns in cleaned dataset:")
for i, col in enumerate(df_cleaned.columns, 1):
    print(f"{i:2d}. {col}")

# Save the cleaned dataset
df_cleaned.to_csv('../data/titanic_cleaned.csv', index=False)
print("\n✓ Cleaned dataset saved as '../data/titanic_cleaned.csv'")

print("\n" + "=" * 60)
print("DATA CLEANING & PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)

# Additional insights
print("\n" + "=" * 40)
print("ADDITIONAL INSIGHTS")
print("=" * 40)

if 'Survived' in df_cleaned.columns:
    survival_rate = df_cleaned['Survived'].mean()
    print(f"Overall survival rate: {survival_rate:.2%}")

print(f"Data reduction: {len(df) - len(df_cleaned)} rows removed ({((len(df) - len(df_cleaned))/len(df)*100):.1f}%)")
print("Dataset is now ready for machine learning modeling!")