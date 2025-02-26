import os
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

warnings.filterwarnings('ignore')

# Ensure the dataset file exists
file_path = 'unsw_nb15_dataset.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' was not found. Please check the file path and try again.")

# Load dataset
df = pd.read_csv(file_path)
df.drop(['id'], axis=1, inplace=True)
df.drop(['label'], axis=1, inplace=True)

# Convert attack category to binary (DoS vs. others)
df['is_ddos'] = df['attack_cat'].apply(lambda x: 1 if x == 'DoS' else 0)
df.drop(['attack_cat'], axis=1, inplace=True)

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
    if df_cat[feature].nunique() > 6:
        df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

# Apply One-Hot Encoding to categorical features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), df.select_dtypes(exclude=[np.number]).columns)], remainder='passthrough')
X = np.array(ct.fit_transform(df.drop(columns=['is_ddos'])))
y = df['is_ddos']

# Feature selection using chi-square test
best_features = SelectKBest(score_func=chi2, k='all')
X = best_features.fit_transform(X, y)

# Standardize features
sc = StandardScaler()
X = sc.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost model
xgb_model = XGBClassifier(
    eval_metric='mlogloss', 
    colsample_bytree=0.95, 
    learning_rate=0.03, 
    max_depth=15, 
    n_estimators=500, 
    subsample=0.95, 
    gamma=0.2, 
    reg_lambda=2.0, 
    reg_alpha=0.5
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
f1s_xgb = f1_score(y_test, y_pred_xgb)

# Get feature names from the one-hot encoder
encoded_feature_names = ct.named_transformers_['encoder'].get_feature_names_out(df.select_dtypes(exclude=[np.number]).columns)
all_feature_names = np.concatenate([encoded_feature_names, df.select_dtypes(include=[np.number]).columns])

# Adjust feature names length to match transformed features
all_feature_names = all_feature_names[:X.shape[1]]

# Feature importance from XGBoost
xgb_importance = pd.DataFrame({'Feature': all_feature_names, 'Importance': xgb_model.feature_importances_})
xgb_importance = xgb_importance.sort_values(by='Importance', ascending=False)

print("\nXGBoost Feature Importance:")
print(xgb_importance)

# Print performance metrics
print("XGBoost Model Performance:")
print("Accuracy: ", "{:.2%}".format(accuracy_xgb))
print("Recall: ", "{:.2%}".format(recall_xgb))
print("Precision: ", "{:.2%}".format(precision_xgb))
print("F1-Score: ", "{:.2%}".format(f1s_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
