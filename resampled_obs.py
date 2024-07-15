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


# Function to calculate Spearman correlations for different metric pairs
def calculate_correlations(df):
    corr_f1_roc, _ = spearmanr(df['F1 Score'], df['ROC AUC'])
    corr_f1_pr, _ = spearmanr(df['F1 Score'], df['PR AUC'])
    corr_roc_pr, _ = spearmanr(df['ROC AUC'], df['PR AUC'])
    corr_f1_kappa, _ = spearmanr(df['F1 Score'], df['Kappa'])
    corr_f1_mcc, _ = spearmanr(df['F1 Score'], df['MCC'])
    corr_mcc_kappa, _ = spearmanr(df['MCC'], df['Kappa'])
    return corr_f1_roc, corr_f1_pr, corr_roc_pr, corr_f1_kappa, corr_f1_mcc, corr_mcc_kappa

# Unique resampling percentages
resampling_percentages = data['Anomaly Percentage'].unique()

# Set up a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10)) # Adjust the size as needed
axs = axs.flatten() # Flatten the 2D array of axes for easy iteration

# Iterate over each resampling percentage and plot
for idx, percentage in enumerate(sorted(resampling_percentages)):
    if idx >= 4: # Only plot for the first four percentages
        break

    subset = data[data['Anomaly Percentage'] == percentage]
    unique_observations = subset['Number of Samples'].unique()

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
        obs_subset = subset[subset['Number of Samples'] == obs]
        corr_values = calculate_correlations(obs_subset)
        for key, value in zip(correlations.keys(), corr_values):
            correlations[key].append(value)

    # Plot on the current subplot axis
    ax = axs[idx]
    for label, values in correlations.items():
        ax.plot(unique_observations, values, label=label, marker='o')
    ax.set_title(f'Percentage: {percentage}%')
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Spearman Correlation Coefficient')
    ax.set_xscale("log")
    ax.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
