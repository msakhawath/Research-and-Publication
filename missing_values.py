import os
import numpy as np
import pandas as pd

# Path to the folder containing the datasets
folder_path = r'C:\Users\sakha\Downloads\ds\ds'  # Replace with the path to your folder


# List to hold the count of missing values for each dataset
missing_values_list = []

# Loop through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.npz'):
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)

        # Count missing values
        missing_x = np.isnan(data['x']).sum()
        missing_tx = np.isnan(data['tx']).sum()
        missing_ty = np.isnan(data['ty']).sum()
        total_missing = missing_x + missing_tx + missing_ty

        # Add the counts to the list
        missing_values_list.append([file, missing_x, missing_tx, missing_ty, total_missing])

# Creating a DataFrame
missing_values_df = pd.DataFrame(missing_values_list, columns=['Dataset', 'Missing in x', 'Missing in tx', 'Missing in ty', 'Total Missing'])

# Print the DataFrame
print(missing_values_df)

# Save to CSV
csv_path = os.path.join(folder_path, 'missing_values_summary.csv')
missing_values_df.to_csv(csv_path, index=False)

print(f"Summary saved to {csv_path}")
