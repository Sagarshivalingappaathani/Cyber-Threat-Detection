import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

# Ignore warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('unsw_nb15_dataset.csv')
df.drop(['id'], axis=1, inplace=True)

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

# Improved Artificial Neural Network (ANN) model
ann_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
ann_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Class weights to handle imbalance
class_weights = {0: 1, 1: 5}  # Adjust based on your dataset

# Train the model
ann_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weights, verbose=1)

# Evaluate the model
y_pred_ann = (ann_model.predict(X_test) > 0.5).astype(int)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
recall_ann = recall_score(y_test, y_pred_ann)
precision_ann = precision_score(y_test, y_pred_ann)
f1s_ann = f1_score(y_test, y_pred_ann)

# Print performance metrics
print("\nImproved Artificial Neural Network (ANN) Model Performance:")
print("Accuracy: ", "{:.2%}".format(accuracy_ann))
print("Recall: ", "{:.2%}".format(recall_ann))
print("Precision: ", "{:.2%}".format(precision_ann))
print("F1-Score: ", "{:.2%}".format(f1s_ann))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ann))