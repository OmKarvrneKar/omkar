<<<<<<< HEAD
# 🚢 Titanic Data Cleaning & Preprocessing

A comprehensive data science project focused on cleaning and preprocessing the famous Titanic dataset for machine learning applications.

## 📋 Project Overview

This project demonstrates professional data cleaning and preprocessing techniques on the Titanic dataset. The goal is to prepare the raw dataset for machine learning models by handling missing values, encoding categorical features, standardizing numerical data, and removing outliers.

## 🎯 Objectives

- **Data Exploration**: Analyze the dataset structure and identify data quality issues
- **Missing Value Handling**: Implement appropriate imputation strategies
- **Feature Engineering**: Convert categorical variables to numerical format
- **Data Standardization**: Scale numerical features for optimal model performance
- **Outlier Detection**: Identify and handle outliers using statistical methods
- **Data Validation**: Ensure the final dataset is ML-ready

## 📁 Project Structure

```
titanic-data-cleaning/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── titanic.csv             # Original dataset
│   └── titanic_cleaned.csv     # Cleaned dataset
│
├── scripts/
│   ├── titanic_data_cleaning.py    # Main data cleaning script
│   └── titanic_model_training.py   # ML model training script
│
└── docs/
    └── USAGE.md                # Detailed usage guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/OmKarvrneKar/omkar.git
   cd omkar
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Run the data cleaning script**
   ```bash
   python scripts/titanic_data_cleaning.py
   ```

2. **Run the model training script**
   ```bash
   python scripts/titanic_model_training.py
   ```

## 📊 Data Cleaning Process

### 1. Initial Data Exploration
- Dataset shape: 891 rows × 12 columns
- Missing values analysis
- Data type identification

### 2. Missing Value Treatment
- **Age**: 177 missing values → Imputed with median (28.0)
- **Cabin**: 687 missing values (77.1%) → Column dropped
- **Embarked**: 2 missing values → Imputed with mode ('S')

### 3. Feature Engineering
- **Sex**: Label encoded (female: 0, male: 1)
- **Embarked**: One-hot encoded (Embarked_C, Embarked_Q, Embarked_S)
- **Removed**: Name, Ticket columns (not useful for ML)

### 4. Data Standardization
- **Age** and **Fare**: Standardized using StandardScaler
- Mean ≈ 0, Standard deviation ≈ 1

### 5. Outlier Removal
- **Method**: Interquartile Range (IQR)
- **Age**: 66 outliers removed
- **Fare**: 107 outliers removed
- **Total**: 173 outliers removed (19.4% data reduction)

## 📈 Results

- **Final dataset**: 718 rows × 11 columns
- **No missing values**: ✅
- **All numerical features**: ✅
- **Standardized data**: ✅
- **ML-ready**: ✅
- **Survival rate**: 33.43%

## 🛠️ Technologies Used

- **Python 3.13**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning tools
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## 📚 Key Features

- ✅ Comprehensive missing value handling
- ✅ Robust outlier detection and removal
- ✅ Professional feature encoding
- ✅ Data standardization
- ✅ Detailed logging and progress tracking
- ✅ Automated data validation
- ✅ Clean, documented code

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Om Karvrnekar**
- GitHub: [@OmKarvrneKar](https://github.com/OmKarvrneKar)
- Repository: [omkar](https://github.com/OmKarvrneKar/omkar)

## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- The data science community for best practices and methodologies
- Contributors and reviewers

---

⭐ **Star this repository if you found it helpful!** ⭐
