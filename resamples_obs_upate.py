import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the dataset
file_path = r'C:\Users\sakha\Downloads\ds\multiple_percentages_all.csv'
data = pd.read_csv(file_path)

# Define 'resampling_percentages' if it's not already defined in your code
resampling_percentages = data['Anomaly Percentage'].unique()

# Function to calculate Spearman correlations for different metric pairs
def calculate_correlations(df):
    corr_f1_roc, _ = spearmanr(df['F1 Score'], df['ROC AUC'])
    corr_f1_pr, _ = spearmanr(df['F1 Score'], df['PR AUC'])
    corr_roc_pr, _ = spearmanr(df['ROC AUC'], df['PR AUC'])
    corr_f1_kappa, _ = spearmanr(df['F1 Score'], df['Kappa'])
    corr_f1_mcc, _ = spearmanr(df['F1 Score'], df['MCC'])
    corr_mcc_kappa, _ = spearmanr(df['MCC'], df['Kappa'])
    return corr_f1_roc, corr_f1_pr, corr_roc_pr, corr_f1_kappa, corr_f1_mcc, corr_mcc_kappa

# Set up a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()

# Iterate over each resampling percentage and plot
for idx, percentage in enumerate(sorted(resampling_percentages)):
    # Only plot for the first four percentages due to subplot grid size
    if idx >= 4:
        break

    # Filter the data for the current percentage
    subset = data[data['Anomaly Percentage'] == percentage]
    # Get unique values of 'Number of Samples' for the current subset
    unique_observations = subset['Number of Samples'].unique()

    # Initialize dictionaries to store correlations
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
        obs_subset = subset[subset['Number of Samples'] == obs].dropna(subset=['F1 Score', 'ROC AUC', 'PR AUC', 'Kappa', 'MCC'])
        if not obs_subset.empty: # Check if there are non-NaN data points
            corr_values = calculate_correlations(obs_subset)
            for key, value in zip(correlations.keys(), corr_values):
                correlations[key].append(value)
        else: # If all data points are NaN, append NaN to maintain the correct index
            for key in correlations.keys():
                correlations[key].append(float('nan'))

    # Plot on the current subplot axis
    ax = axs[idx]
    for label, values in correlations.items():
        # Plot and connect only non-NaN values
        valid_indices = [i for i, v in enumerate(values) if not pd.isna(v)]
        valid_values = [v for v in values if not pd.isna(v)]
        valid_observations = [unique_observations[i] for i in valid_indices]
        ax.plot(valid_observations, valid_values, label=label, marker='o')
    ax.set_title(f'Percentage: {percentage}%')
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Spearman Correlation Coefficient')
    #ax.set_xscale("log")
    ax.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
