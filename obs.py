import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the dataset
file_path = r'C:\Users\sakha\Downloads\ds\multiple_percentages_all.csv'
data = pd.read_csv(file_path)

columns_to_drop = data.columns[-3:]

# Drop these columns from the DataFrame
data = data.drop(columns=columns_to_drop)
print(data.columns)

# Filter for only 50% Initial Anomaly Percentage
data = data[data['Initial Anomaly Percentage'] == 50]

# Function to calculate Spearman correlations for different metric pairs
def calculate_correlations(df):
    corr_f1_roc, _ = spearmanr(df['F1 Score'], df['ROC AUC'])
    corr_f1_pr, _ = spearmanr(df['F1 Score'], df['PR AUC'])
    corr_roc_pr, _ = spearmanr(df['ROC AUC'], df['PR AUC'])
    corr_f1_kappa, _ = spearmanr(df['F1 Score'], df['Kappa'])
    corr_f1_mcc, _ = spearmanr(df['F1 Score'], df['MCC'])
    corr_mcc_kappa, _ = spearmanr(df['MCC'], df['Kappa'])
    return corr_f1_roc, corr_f1_pr, corr_roc_pr, corr_f1_kappa, corr_f1_mcc, corr_mcc_kappa

# Unique number of samples
unique_observations = data['Number of Samples'].unique()

# Initialize lists to store correlations
correlations = {
    'F1 vs ROC': [],
    'F1 vs PR': [],
    'ROC vs PR': [],
    'F1 vs Kappa': [],
    'F1 vs MCC': [],
    'MCC vs Kappa': []
}

# Calculate correlations for each number of observations
for obs in sorted(unique_observations):
    obs_subset = data[data['Number of Samples'] == obs]
    corr_values = calculate_correlations(obs_subset)
    for key, value in zip(correlations.keys(), corr_values):
        correlations[key].append(value)

# Create plot
plt.figure(figsize=(12, 8))
for label, values in correlations.items():
    plt.plot(unique_observations, values, label=label, marker='o')
plt.xticks(rotation=45)
plt.title('Spearman Correlation')
plt.ylabel('Spearman Correlation Coefficient')
plt.xlabel('Number of Observations')
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show()
