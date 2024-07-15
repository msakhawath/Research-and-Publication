import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    corr_f1_MCC, _ = spearmanr(df['F1 Score'], df['MCC'])
    corr_f1_Kappa, _ = spearmanr(df['F1 Score'], df['Kappa'])
    corr_MCC_Kappa, _ = spearmanr(df['MCC'], df['Kappa'])
    return corr_f1_roc, corr_f1_pr, corr_roc_pr, corr_f1_MCC, corr_f1_Kappa, corr_MCC_Kappa

# Unique resampling percentages
resampling_percentages = data['Anomaly Percentage'].unique()

# Prepare a figure for the plots
plt.figure(figsize=(15, 15))

# Loop over each resampling percentage
for i, percentage in enumerate(sorted(resampling_percentages), start=1):
    subset = data[data['Anomaly Percentage'] == percentage]
    unique_models = subset['Model'].unique()

    # Initialize lists to store correlations
    correlations_f1_roc = []
    correlations_f1_pr = []
    correlations_roc_pr = []
    correlations_f1_kappa = []
    correlations_f1_MCC = []
    correlations_MCC_Kappa = []

    # Calculate correlations for each model
    for model in unique_models:
        model_subset = subset[subset['Model'] == model]
        corr_f1_roc, corr_f1_pr, corr_roc_pr, corr_f1_MCC, corr_f1_Kappa, corr_MCC_Kappa = calculate_correlations(model_subset)
        correlations_f1_roc.append(corr_f1_roc)
        correlations_f1_pr.append(corr_f1_pr)
        correlations_roc_pr.append(corr_roc_pr)
        correlations_f1_kappa.append(corr_f1_Kappa)
        correlations_f1_MCC.append(corr_f1_MCC)
        correlations_MCC_Kappa.append(corr_MCC_Kappa)

    # Create subplot for current percentage
    plt.subplot(2, 2, i)
    plt.plot(unique_models, correlations_f1_roc, label='F1 vs ROC', color='blue', marker='o')
    plt.plot(unique_models, correlations_f1_pr, label='F1 vs PR', color='red', marker='o')
    plt.plot(unique_models, correlations_roc_pr, label='ROC vs PR', color='green', marker='o')
    plt.plot(unique_models, correlations_f1_kappa, label='F1 vs Kappa', color='purple', marker='o')
    plt.plot(unique_models, correlations_f1_MCC, label='F1 vs MCC', color='yellow', marker='o')
    plt.plot(unique_models, correlations_MCC_Kappa, label='MCC vs Kappa', color='black', marker='o')
    plt.xticks(rotation=45)
    plt.title(f'Spearman Correlation for Resampling Percentage: {percentage}%')
    plt.ylabel('Spearman Correlation Coefficient')
    #plt.xlabel('Models')
    plt.legend()

plt.tight_layout()
plt.show()
