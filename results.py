import pandas as pd
import matplotlib.pyplot as plt

# Data for XGBoost Models
xgboost_data = {
    "Attack Category": ["DOS", "Fuzzers", "Generic", "Analysis", "Reconnaissance", "Backdoor", "Shellcode", "Worms", "Normal"],
    "Accuracy (%)": [95.54, 97.72, 99.48, 97.45, 98.64, 97.96, 99.78, 99.96, 100.00],
    "Recall (%)": [97.86, 98.83, 99.29, 99.60, 98.18, 99.88, 99.96, 99.99, 100.00],
    "Precision (%)": [93.53, 96.68, 99.67, 95.50, 99.10, 96.19, 99.60, 99.93, 100.00],
    "F1-Score (%)": [95.64, 97.74, 99.48, 97.50, 98.64, 98.00, 99.78, 99.96, 100.00]
}

# Data for Random Forest Model
random_forest_data = {
    "Attack Category": ["DOS", "Fuzzers", "Generic", "Analysis", "Reconnaissance", "Backdoor", "Shellcode", "Worms", "Normal"],
    "Accuracy (%)": [94.55, 97.69, 99.50, 97.52, 98.55, 98.02, 99.81, 99.98, 100.00],
    "Recall (%)": [99.29, 98.81, 99.16, 99.68, 98.09, 99.90, 99.99, 99.98, 100.00],
    "Precision (%)": [90.70, 96.65, 99.83, 95.55, 99.01, 96.27, 99.64, 99.98, 100.00],
    "F1-Score (%)": [94.80, 97.72, 99.49, 97.57, 98.55, 98.05, 99.81, 99.98, 100.00]
}

# Convert to DataFrames
df_xgboost = pd.DataFrame(xgboost_data)
df_random_forest = pd.DataFrame(random_forest_data)

def save_table_as_image(df, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Add title
    plt.title(title, fontsize=14, fontweight="bold")

    # Save figure
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Generate and save the tables as images
save_table_as_image(df_xgboost, "XGBoost Model Performance", "xgboost_performance.png")
save_table_as_image(df_random_forest, "Random Forest Model Performance", "random_forest_performance.png")

print("Tables saved as 'xgboost_performance.png' and 'random_forest_performance.png'")
