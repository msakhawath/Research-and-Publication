import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the dataset
file_path = r'C:\Users\sakha\Downloads\ds\multiple_percentages_all.csv'
data = pd.read_csv(file_path)

# Function to calculate Spearman correlations for different metric pairs
def calculate_correlations(df):
    corr_f1_roc, _ = spearmanr(df['F1 Score'], df['ROC AUC'])
    corr_f1_pr, _ = spearmanr(df['F1 Score'], df['PR AUC'])
    corr_roc_pr, _ = spearmanr(df['ROC AUC'], df['PR AUC'])
    return corr_f1_roc, corr_f1_pr, corr_roc_pr

# Unique resampling percentages
resampling_percentages = data['Anomaly Percentage'].unique()

# Prepare individual figures for each resampling percentage
for percentage in sorted(resampling_percentages):
    subset = data[data['Anomaly Percentage'] == percentage]
    unique_datasets = subset['Dataset Name'].unique()

    # Initialize lists to store correlations
    correlations_f1_roc = []
    correlations_f1_pr = []
    correlations_roc_pr = []

    # Calculate correlations for each dataset
    for dataset in unique_datasets:
        dataset_subset = subset[subset['Dataset Name'] == dataset]
        corr_f1_roc, corr_f1_pr, corr_roc_pr = calculate_correlations(dataset_subset)
        correlations_f1_roc.append(corr_f1_roc)
        correlations_f1_pr.append(corr_f1_pr)
        correlations_roc_pr.append(corr_roc_pr)

    # Create plot for current percentage
    plt.figure(figsize=(10, 6))
    plt.plot(unique_datasets, correlations_f1_roc, label='F1 vs ROC', color='blue', marker='o')
    plt.plot(unique_datasets, correlations_f1_pr, label='F1 vs PR', color='red', marker='o')
    plt.plot(unique_datasets, correlations_roc_pr, label='ROC vs PR', color='green', marker='o')
    plt.xticks(rotation=45)
    plt.title(f'Spearman Correlation for Resampling Percentage: {percentage}%')
    plt.ylabel('Spearman Correlation Coefficient')
    plt.xlabel('Datasets')
    plt.legend()
    plt.tight_layout()
    plt.show()
