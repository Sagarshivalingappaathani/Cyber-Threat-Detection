import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('unsw_nb15_dataset.csv')
df.drop(['id'], axis=1, inplace=True)

# Keep the attack category as is for multi-classifications
# Check for any missing values in attack_cat and replace with 'normal' if needed
df['attack_cat'] = df['attack_cat'].fillna('normal')

# Print distribution of attack categories
print("Attack Category Distribution:")
print(df['attack_cat'].value_counts())

# Handle numerical features (clamping extreme values and applying log transformation)
df_numeric = df.select_dtypes(include=[np.number])
for feature in df_numeric.columns:
    if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
        df[feature] = np.where(df[feature] < df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

for feature in df_numeric.columns:
    if df_numeric[feature].nunique() > 50:
        df[feature] = np.log(df[feature] + 1) if df_numeric[feature].min() == 0 else np.log(df[feature])

# Handle categorical features by limiting unique values and encoding
df_cat = df.select_dtypes(exclude=[np.number])
for feature in df_cat.columns:
    if feature != 'attack_cat' and df_cat[feature].nunique() > 6:  # Don't limit attack_cat values
        df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

# Encode the target variable (attack_cat)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['attack_cat'])

# Store mapping for later interpretation
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:")
for category, label in label_mapping.items():
    print(f"{label}: {category}")

# Apply One-Hot Encoding to categorical features (excluding attack_cat)
categorical_cols = [col for col in df.select_dtypes(exclude=[np.number]).columns if col != 'attack_cat']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols)], remainder='passthrough')
X = np.array(ct.fit_transform(df.drop(columns=['attack_cat'])))

# Feature selection using chi-square test
best_features = SelectKBest(score_func=chi2, k='all')
X = best_features.fit_transform(X, y)

# Standardize features
sc = StandardScaler()
X = sc.fit_transform(X)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost model with multi-class configuration
print("Training XGBoost model...")
start_time = time.time()
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    eval_metric='mlogloss',
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=12,
    n_estimators=400,
    subsample=0.8,
    gamma=0.1,
    reg_lambda=1.0,
    reg_alpha=0.5,
    use_label_encoder=False,
    tree_method='hist'  # Can be faster for large datasets
)
xgb_model.fit(X_train, y_train)
xgb_training_time = time.time() - start_time
print(f"XGBoost training completed in {xgb_training_time:.2f} seconds")

y_pred_xgb = xgb_model.predict(X_test)

# Train Random Forest model
print("Training Random Forest model...")
start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1  # Use all available cores
)
rf_model.fit(X_train, y_train)
rf_training_time = time.time() - start_time
print(f"Random Forest training completed in {rf_training_time:.2f} seconds")

y_pred_rf = rf_model.predict(X_test)

# Evaluate XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted')
f1_weighted_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_weighted_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Print performance metrics for both models
print("\nXGBoost Model Performance:")
print("Accuracy: ", "{:.2%}".format(accuracy_xgb))
print("Precision: ", "{:.2%}".format(precision_xgb))
print("Recall: ", "{:.2%}".format(recall_xgb))
print("F1-Score (Weighted): ", "{:.2%}".format(f1_weighted_xgb))
print("Training Time: ", "{:.2f} seconds".format(xgb_training_time))

print("\nRandom Forest Model Performance:")
print("Accuracy: ", "{:.2%}".format(accuracy_rf))
print("Precision: ", "{:.2%}".format(precision_rf))
print("Recall: ", "{:.2%}".format(recall_rf))
print("F1-Score (Weighted): ", "{:.2%}".format(f1_weighted_rf))
print("Training Time: ", "{:.2f} seconds".format(rf_training_time))


# Create confusion matrices with proper labels
def plot_confusion_matrix(y_true, y_pred, model_name, class_names):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix for {model_name}:")
    # Print header
    header = "True\\Pred"
    for name in class_names:
        header += f"\t{name[:4]}"  # Abbreviate class names for better display
    print(header)
    
    # Print each row
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:4]}"  # Abbreviate class name
        for val in row:
            row_str += f"\t{val}"
        print(row_str)
    
    return cm

# Plot confusion matrices
xgb_cm = plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost", le.classes_)
rf_cm = plot_confusion_matrix(y_test, y_pred_rf, "Random Forest", le.classes_)

