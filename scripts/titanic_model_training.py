# Task: AI/ML Internship Task 1 - Model Training & Evaluation
# Dataset: Cleaned Titanic Data (titanic_cleaned.csv)
# Objective: To train a classification model to predict passenger survival and evaluate its performance.

# Step 1: Import Libraries and Load Cleaned Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TITANIC SURVIVAL PREDICTION - MODEL TRAINING & EVALUATION")
print("=" * 70)

# Load the cleaned dataset
try:
    df_cleaned = pd.read_csv('../data/titanic_cleaned.csv')
    print("✓ Cleaned dataset loaded successfully!")
    print(f"Dataset shape: {df_cleaned.shape}")
except FileNotFoundError:
    print("❌ Error: 'titanic_cleaned.csv' not found in the data directory.")
    print("Please ensure you have run the data cleaning script first.")
    exit()

# Display basic information about the cleaned dataset
print("\n" + "=" * 50)
print("STEP 1: DATASET OVERVIEW")
print("=" * 50)

print(f"\nDataset shape: {df_cleaned.shape}")
print(f"Columns: {list(df_cleaned.columns)}")
print(f"Missing values: {df_cleaned.isnull().sum().sum()}")

print("\nFirst 5 rows of cleaned data:")
print(df_cleaned.head())

print("\nDataset info:")
print(df_cleaned.info())

print("\nTarget variable distribution:")
survival_counts = df_cleaned['Survived'].value_counts()
print(survival_counts)
print(f"Survival rate: {df_cleaned['Survived'].mean():.2%}")

# Step 2: Define Features (X) and Target (y)
print("\n" + "=" * 50)
print("STEP 2: DEFINE FEATURES AND TARGET")
print("=" * 50)

# Set target variable
y = df_cleaned['Survived']
print(f"✓ Target variable (y): 'Survived'")
print(f"Target shape: {y.shape}")

# Set feature variables (drop Survived and PassengerId)
columns_to_drop = ['Survived', 'PassengerId']
X = df_cleaned.drop(columns=columns_to_drop)
print(f"✓ Feature variables (X): {list(X.columns)}")
print(f"Features shape: {X.shape}")

print(f"\nFeature types:")
for col in X.columns:
    print(f"  {col}: {X[col].dtype}")

# Step 3: Split Data into Training and Testing Sets
print("\n" + "=" * 50)
print("STEP 3: TRAIN-TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Data split completed with test_size=0.2 and random_state=42")
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

print(f"\nTarget distribution in training set:")
train_survival = y_train.value_counts(normalize=True)
print(train_survival)

print(f"\nTarget distribution in testing set:")
test_survival = y_test.value_counts(normalize=True)
print(test_survival)

# Step 4: Train and Evaluate a Logistic Regression Model
print("\n" + "=" * 50)
print("STEP 4: LOGISTIC REGRESSION MODEL")
print("=" * 50)

# Initialize and train Logistic Regression
print("Training Logistic Regression model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
print("✓ Logistic Regression model trained successfully")

# Make predictions
lr_predictions = lr_model.predict(X_test)
lr_probabilities = lr_model.predict_proba(X_test)[:, 1]

# Calculate accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"\n📊 LOGISTIC REGRESSION RESULTS:")
print(f"Accuracy: {lr_accuracy:.4f} ({lr_accuracy:.2%})")

# Detailed classification report
print(f"\nDetailed Classification Report:")
lr_report = classification_report(y_test, lr_predictions)
print(lr_report)

# Confusion Matrix
lr_cm = confusion_matrix(y_test, lr_predictions)
print(f"Confusion Matrix:")
print(lr_cm)

# ROC AUC Score
lr_roc_auc = roc_auc_score(y_test, lr_probabilities)
print(f"ROC AUC Score: {lr_roc_auc:.4f}")

# Step 5: Train and Evaluate a Random Forest Model
print("\n" + "=" * 50)
print("STEP 5: RANDOM FOREST MODEL")
print("=" * 50)

# Initialize and train Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✓ Random Forest model trained successfully")

# Make predictions
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"\n📊 RANDOM FOREST RESULTS:")
print(f"Accuracy: {rf_accuracy:.4f} ({rf_accuracy:.2%})")

# Detailed classification report
print(f"\nDetailed Classification Report:")
rf_report = classification_report(y_test, rf_predictions)
print(rf_report)

# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_predictions)
print(f"Confusion Matrix:")
print(rf_cm)

# ROC AUC Score
rf_roc_auc = roc_auc_score(y_test, rf_probabilities)
print(f"ROC AUC Score: {rf_roc_auc:.4f}")

# Model Comparison
print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)

comparison_data = {
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [f'{lr_accuracy:.4f}', f'{rf_accuracy:.4f}'],
    'ROC AUC': [f'{lr_roc_auc:.4f}', f'{rf_roc_auc:.4f}']
}
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Determine best model
if rf_accuracy > lr_accuracy:
    best_model = "Random Forest"
    best_accuracy = rf_accuracy
else:
    best_model = "Logistic Regression"
    best_accuracy = lr_accuracy

print(f"\n🏆 Best performing model: {best_model}")
print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")

# Feature Importance Analysis (Random Forest)
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Random Forest Feature Importance:")
print(feature_importance.to_string(index=False))

# Visualizations
print("\n" + "=" * 50)
print("CREATING VISUALIZATIONS")
print("=" * 50)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Titanic Survival Prediction - Model Analysis', fontsize=16, fontweight='bold')

# 1. Confusion Matrix - Logistic Regression
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
axes[0,0].set_title('Logistic Regression\nConfusion Matrix')
axes[0,0].set_xlabel('Predicted')
axes[0,0].set_ylabel('Actual')

# 2. Confusion Matrix - Random Forest
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[0,1])
axes[0,1].set_title('Random Forest\nConfusion Matrix')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Actual')

# 3. Model Accuracy Comparison
models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_accuracy, rf_accuracy]
colors = ['skyblue', 'lightgreen']
bars = axes[0,2].bar(models, accuracies, color=colors)
axes[0,2].set_title('Model Accuracy Comparison')
axes[0,2].set_ylabel('Accuracy')
axes[0,2].set_ylim(0, 1)
# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracies):
    axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{acc:.3f}', ha='center', va='bottom')

# 4. Feature Importance
top_features = feature_importance.head(8)  # Top 8 features
axes[1,0].barh(top_features['Feature'], top_features['Importance'], color='coral')
axes[1,0].set_title('Random Forest\nTop Feature Importance')
axes[1,0].set_xlabel('Importance')

# 5. ROC Curves
# Logistic Regression ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probabilities)
axes[1,1].plot(lr_fpr, lr_tpr, color='blue', lw=2, 
               label=f'Logistic Regression (AUC = {lr_roc_auc:.3f})')

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)
axes[1,1].plot(rf_fpr, rf_tpr, color='green', lw=2, 
               label=f'Random Forest (AUC = {rf_roc_auc:.3f})')

# Diagonal line
axes[1,1].plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random Classifier')
axes[1,1].set_xlim([0.0, 1.0])
axes[1,1].set_ylim([0.0, 1.05])
axes[1,1].set_xlabel('False Positive Rate')
axes[1,1].set_ylabel('True Positive Rate')
axes[1,1].set_title('ROC Curves Comparison')
axes[1,1].legend(loc="lower right")
axes[1,1].grid(True, alpha=0.3)

# 6. Survival Rate by Key Features
# Create a simple analysis plot
survival_by_sex = df_cleaned.groupby('Sex_encoded')['Survived'].mean()
sex_labels = ['Female', 'Male']
axes[1,2].bar(sex_labels, survival_by_sex.values, color=['pink', 'lightblue'])
axes[1,2].set_title('Survival Rate by Gender')
axes[1,2].set_ylabel('Survival Rate')
axes[1,2].set_ylim(0, 1)
for i, v in enumerate(survival_by_sex.values):
    axes[1,2].text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Model analysis visualizations saved as 'model_analysis_visualization.png'")

# Additional Model Insights
print("\n" + "=" * 50)
print("ADDITIONAL MODEL INSIGHTS")
print("=" * 50)

# Logistic Regression Coefficients
print("Logistic Regression Coefficients (Top 5 by absolute value):")
lr_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
})
lr_coef['Abs_Coefficient'] = abs(lr_coef['Coefficient'])
lr_coef_sorted = lr_coef.sort_values('Abs_Coefficient', ascending=False)
print(lr_coef_sorted.head().to_string(index=False))

# Model Performance Summary
print(f"\n📈 FINAL PERFORMANCE SUMMARY:")
print(f"Dataset: {df_cleaned.shape[0]} samples, {X.shape[1]} features")
print(f"Train/Test Split: {len(X_train)}/{len(X_test)} samples")
print(f"Baseline (always predict majority class): {max(y_test.value_counts())/len(y_test):.3f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.3f} (Improvement: +{lr_accuracy - max(y_test.value_counts())/len(y_test):.3f})")
print(f"Random Forest Accuracy: {rf_accuracy:.3f} (Improvement: +{rf_accuracy - max(y_test.value_counts())/len(y_test):.3f})")

# Save model results
results_summary = {
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_accuracy, rf_accuracy],
    'ROC_AUC': [lr_roc_auc, rf_roc_auc],
    'Training_Size': [len(X_train), len(X_train)],
    'Testing_Size': [len(X_test), len(X_test)]
}
results_df = pd.DataFrame(results_summary)
results_df.to_csv('model_results_summary.csv', index=False)
print(f"\n✓ Model results saved as 'model_results_summary.csv'")

print("\n" + "=" * 70)
print("MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("🎯 Ready for deployment or further model improvement!")