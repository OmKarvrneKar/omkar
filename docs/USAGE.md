# Usage Guide - Titanic Data Cleaning Project

This guide provides detailed instructions on how to use the Titanic data cleaning scripts and understand the project workflow.

## 📚 Table of Contents
1. [Quick Start](#quick-start)
2. [Script Details](#script-details)
3. [Data Cleaning Process](#data-cleaning-process)
4. [Output Files](#output-files)
5. [Customization](#customization)
6. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites
Ensure you have Python 3.8+ installed and all dependencies from `requirements.txt`.

### Basic Usage
```bash
# 1. Navigate to project directory
cd path/to/titanic-data-cleaning

# 2. Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Run data cleaning
python scripts/titanic_data_cleaning.py

# 4. Run model training (optional)
python scripts/titanic_model_training.py
```

## 📋 Script Details

### 1. `titanic_data_cleaning.py`

**Purpose**: Complete data preprocessing pipeline for the Titanic dataset.

**Key Features**:
- Automatic missing value detection and handling
- Categorical variable encoding
- Numerical feature standardization
- Outlier detection and removal
- Data validation and quality checks

**Input**: `data/titanic.csv`
**Output**: `data/titanic_cleaned.csv`

**Process Flow**:
```
Raw Data → Missing Values → Encoding → Standardization → Outliers → Clean Data
```

### 2. `titanic_model_training.py`

**Purpose**: Train machine learning models on the cleaned dataset.

**Features**:
- Multiple ML algorithms
- Model evaluation and comparison
- Performance metrics
- Cross-validation

**Input**: `data/titanic_cleaned.csv`
**Output**: Model results and performance metrics

## 🔧 Data Cleaning Process

### Step 1: Initial Exploration
```python
# What happens:
- Load dataset using pandas
- Display basic information (shape, columns, data types)
- Identify missing values
- Generate summary statistics
```

**Output Example**:
```
Dataset shape: (891, 12)
Missing values:
- Age: 177 (19.9%)
- Cabin: 687 (77.1%)
- Embarked: 2 (0.2%)
```

### Step 2: Missing Value Handling

| Column | Strategy | Reason |
|--------|----------|--------|
| **Age** | Median imputation | Central tendency, robust to outliers |
| **Cabin** | Drop column | Too many missing values (77.1%) |
| **Embarked** | Mode imputation | Categorical variable, most common value |

### Step 3: Feature Engineering

**Categorical Encoding**:
- **Sex**: Label Encoding (Female: 0, Male: 1)
- **Embarked**: One-Hot Encoding (3 binary columns)

**Feature Removal**:
- **Name**: Not useful for survival prediction
- **Ticket**: Too many unique values, no clear pattern

### Step 4: Data Standardization

```python
# Applied to numerical features
StandardScaler(): mean = 0, std = 1

Features standardized:
- Age
- Fare
```

### Step 5: Outlier Detection

**Method**: Interquartile Range (IQR)
```python
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

**Results**:
- Age outliers removed: 66
- Fare outliers removed: 107
- Total data reduction: 19.4%

## 📊 Output Files

### `data/titanic_cleaned.csv`
- **Shape**: 718 rows × 11 columns
- **No missing values**: ✅
- **All numerical features**: ✅
- **Standardized data**: ✅

**Column Structure**:
```
1. PassengerId    - Unique identifier
2. Survived       - Target variable (0/1)
3. Pclass         - Passenger class (1/2/3)
4. Age            - Standardized age
5. SibSp          - Siblings/spouses aboard
6. Parch          - Parents/children aboard
7. Fare           - Standardized fare
8. Sex_encoded    - Gender (0=Female, 1=Male)
9. Embarked_C     - Embarked from Cherbourg (0/1)
10. Embarked_Q    - Embarked from Queenstown (0/1)
11. Embarked_S    - Embarked from Southampton (0/1)
```

## ⚙️ Customization

### Modify Imputation Strategy
```python
# In titanic_data_cleaning.py, line ~45
# Change median to mean for Age imputation
age_mean = df['Age'].mean()
df['Age'].fillna(age_mean, inplace=True)
```

### Adjust Outlier Sensitivity
```python
# In remove_outliers_iqr function, line ~135
# Change multiplier from 1.5 to 2.0 for less aggressive outlier removal
lower_bound = Q1 - 2.0 * IQR  # Instead of 1.5
upper_bound = Q3 + 2.0 * IQR  # Instead of 1.5
```

### Add New Features
```python
# Example: Create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Example: Create title feature from names
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
```

## 🐛 Troubleshooting

### Common Issues

**1. FileNotFoundError: 'titanic.csv' not found**
```bash
# Solution: Ensure the dataset is in the data/ directory
# Download from: https://www.kaggle.com/c/titanic/data
```

**2. ModuleNotFoundError: No module named 'pandas'**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**3. Permission Error when saving files**
```bash
# Solution: Check write permissions or run as administrator
# Ensure the output directory exists
```

**4. Memory Error with large datasets**
```python
# Solution: Process data in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    # Process each chunk
```

### Debug Mode

To run with detailed output:
```python
# Add at the beginning of the script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

For large datasets:
```python
# Use efficient data types
df = df.astype({
    'PassengerId': 'int32',
    'Survived': 'int8',
    'Pclass': 'int8'
})
```

## 📈 Expected Results

After running the cleaning script, you should see:

```
✓ Dataset loaded successfully! (891, 12)
✓ Age column imputed with median value: 28.0
✓ Embarked column imputed with mode value: S
✓ Cabin column dropped
✓ Sex column label encoded
✓ Embarked column one-hot encoded
✓ Numerical features standardized
✓ Outliers removed: 173 total
✓ Final dataset: (718, 11) - Ready for ML!
```

## 🔗 Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Data Cleaning Best Practices](https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b)

## 📞 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review the script output for error messages
3. Ensure all dependencies are correctly installed
4. Verify input data format matches expected structure

---

Happy Data Cleaning! 🧹✨