import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('unsw_nb15_dataset.csv')
df.drop(['id'], axis=1, inplace=True)

# Keep the attack category as is for multi-classification
# Check for any missing values in attack_cat and replace with 'normal' if needed
df['attack_cat'] = df['attack_cat'].fillna('normal')

# Print distribution of attack categories
print("Attack Category Distribution:")
print(df['attack_cat'].value_counts())

# Visualize attack distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=df['attack_cat'], order=df['attack_cat'].value_counts().index)
plt.title('Distribution of Attack Categories')
plt.tight_layout()
plt.savefig('attack_distribution.png')
plt.close()

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

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create a dictionary to store model results
model_results = {
    'Model': [],
    'Accuracy': [],
    'Precision (Weighted)': [],
    'Recall (Weighted)': [],
    'F1 (Weighted)': [],
    'F1 (Macro)': [],
    'MCC': [],
    'Training Time': []
}

# Function to evaluate and store result
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    print(f"Training {name} model...")
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"{name} training completed in {training_time:.2f} seconds")
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Store results
    model_results['Model'].append(name)
    model_results['Accuracy'].append(accuracy)
    model_results['Precision (Weighted)'].append(precision_weighted)
    model_results['Recall (Weighted)'].append(recall_weighted)
    model_results['F1 (Weighted)'].append(f1_weighted)
    model_results['F1 (Macro)'].append(f1_macro)
    model_results['MCC'].append(mcc)
    model_results['Training Time'].append(training_time)
    
    print(f"\n{name} Model Performance:")
    print("Accuracy: ", "{:.2%}".format(accuracy))
    print("Precision (Weighted): ", "{:.2%}".format(precision_weighted))
    print("Recall (Weighted): ", "{:.2%}".format(recall_weighted))
    print("F1-Score (Weighted): ", "{:.2%}".format(f1_weighted))
    print("F1-Score (Macro): ", "{:.2%}".format(f1_macro))
    print("Matthews Correlation Coefficient: ", "{:.4f}".format(mcc))
    print("Training Time: ", "{:.2f} seconds".format(training_time))
    
    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[label_mapping[i] for i in sorted(label_mapping.keys())],
                yticklabels=[label_mapping[i] for i in sorted(label_mapping.keys())])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[label_mapping[i] for i in sorted(label_mapping.keys())]))
    
    return model

# Define and train multiple classification models

# 1. XGBoost
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    eval_metric='mlogloss',
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=12,
    n_estimators=200,
    subsample=0.8,
    gamma=0.1,
    reg_lambda=1.0,
    reg_alpha=0.5,
    use_label_encoder=False,
    tree_method='hist'
)
xgb_model = evaluate_model("XGBoost", xgb_model, X_train, y_train, X_test, y_test)

# 2. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model = evaluate_model("Random Forest", rf_model, X_train, y_train, X_test, y_test)

# 3. Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)
gb_model = evaluate_model("Gradient Boosting", gb_model, X_train, y_train, X_test, y_test)

# 4. Support Vector Machine
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42,
    decision_function_shape='ovr'
)
svm_model = evaluate_model("SVM", svm_model, X_train, y_train, X_test, y_test)

# 5. K-Nearest Neighbors
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    algorithm='auto',
    n_jobs=-1
)
knn_model = evaluate_model("K-Nearest Neighbors", knn_model, X_train, y_train, X_test, y_test)

# 6. Neural Network (MLP)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=200,
    random_state=42
)
mlp_model = evaluate_model("Neural Network (MLP)", mlp_model, X_train, y_train, X_test, y_test)

# 7. Logistic Regression
log_reg_model = LogisticRegression(
    C=1.0,
    solver='saga',  # Good for large datasets
    penalty='l2',
    multi_class='multinomial',
    max_iter=200,
    n_jobs=-1,
    random_state=42
)
log_reg_model = evaluate_model("Logistic Regression", log_reg_model, X_train, y_train, X_test, y_test)

# 8. Naive Bayes
nb_model = GaussianNB()
nb_model = evaluate_model("Naive Bayes", nb_model, X_train, y_train, X_test, y_test)

# 9. Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=15,
    criterion='gini',
    class_weight='balanced',
    random_state=42
)
dt_model = evaluate_model("Decision Tree", dt_model, X_train, y_train, X_test, y_test)

# 10. AdaBoost
ada_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=4),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada_model = evaluate_model("AdaBoost", ada_model, X_train, y_train, X_test, y_test)

# 11. Voting Classifier (Ensemble of multiple models)
voting_model = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ],
    voting='soft'
)
voting_model = evaluate_model("Voting Ensemble", voting_model, X_train, y_train, X_test, y_test)

# Create a results dataframe and save to CSV
results_df = pd.DataFrame(model_results)
results_df.to_csv('model_comparison_results.csv', index=False)

# Sort by F1 (Weighted) score for display
results_df_sorted = results_df.sort_values(by='F1 (Weighted)', ascending=False)
print("\nModel Comparison (Sorted by F1 Weighted Score):")
print(results_df_sorted)

# Visualize results
plt.figure(figsize=(14, 10))

metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)']
for i, metric in enumerate(metrics):
    plt.subplot(3, 1, i+1)
    sns.barplot(x='Model', y=metric, data=results_df_sorted)
    plt.title(f'Model Comparison - {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Find the best model based on weighted F1-score
best_model_idx = results_df['F1 (Weighted)'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\nBest performing model based on F1 (Weighted): {best_model_name}")

# Save feature importance for interpretable models
def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Get indices of top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
        plt.close()

# Plot feature importance for tree-based models
plot_feature_importance(rf_model, "Random Forest")
plot_feature_importance(xgb_model, "XGBoost")
plot_feature_importance(gb_model, "Gradient Boosting")

print("\nAnalysis completed. All results saved to files.")