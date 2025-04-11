import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import time
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

#warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('unsw_nb15_dataset.csv')
df.drop(['id', 'attack_cat'], axis=1, inplace=True)

df_numeric = df.select_dtypes(include=[np.number])
for feature in df_numeric.columns:
    if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
        df[feature] = np.where(df[feature] < df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

df_numeric = df.select_dtypes(include=[np.number])
for feature in df_numeric.columns:
    if df_numeric[feature].nunique() > 50:
        df[feature] = np.log(df[feature] + 1) if df_numeric[feature].min() == 0 else np.log(df[feature])

df_cat = df.select_dtypes(exclude=[np.number])
for feature in df_cat.columns:
    if df_cat[feature].nunique() > 6:
        df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

best_features = SelectKBest(score_func=chi2, k='all')
X = df.iloc[:, 4:-2]
y = df.iloc[:, -1]
best_features.fit(X, y)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

sc = StandardScaler()
X[:, 18:] = sc.fit_transform(X[:, 18:])

smote = SMOTE()
X, y = smote.fit_resample(X, y)

best_model = XGBClassifier(eval_metric='mlogloss', colsample_bytree=0.8, learning_rate=0.1, max_depth=9, n_estimators=200, subsample=0.8)
y_predictions = cross_val_predict(best_model, X, y, cv=5)

accuracy = cross_val_score(best_model, X, y, cv=5, scoring='accuracy').mean()
recall = cross_val_score(best_model, X, y, cv=5, scoring='recall_weighted').mean()
precision = cross_val_score(best_model, X, y, cv=5, scoring='precision_weighted').mean()
f1s = cross_val_score(best_model, X, y, cv=5, scoring='f1_weighted').mean()

print("Cross-validation Accuracy: ", "{:.2%}".format(accuracy))
print("Cross-validation Recall: ", "{:.2%}".format(recall))
print("Cross-validation Precision: ", "{:.2%}".format(precision))
print("Cross-validation F1-Score: ", "{:.2%}".format(f1s))

cm = confusion_matrix(y, y_predictions)
print("Confusion Matrix:\n", cm)
