import os
import numpy as np
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score
from pyod.utils.utility import standardizer

folder_path = r'C:\Users\sakha\Downloads\ds\all dataset'  # Replace with the path to your datasets folder
file_extension = '.npz'

results = []

# Loop over each file in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(file_extension):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)

        X_train = data['x']
        X_test = data['tx']
        y_test = data['ty']

        # Count missing values
        missing_values_train = np.isnan(X_train).sum()
        missing_values_test = np.isnan(X_test).sum()
        missing_values_labels = np.isnan(y_test).sum()
        total_missing_values = missing_values_train + missing_values_test + missing_values_labels

        # Remove missing values
        X_train = X_train[~np.isnan(X_train).any(axis=1)]
        X_test = X_test[~np.isnan(X_test).any(axis=1)]
        y_test = y_test[~np.isnan(y_test)]

        # Standardize the data
        X_train_scaled, X_test_scaled = standardizer(X_train, X_test)

        # Set the contamination rate
        fixed_contamination = 0.1

        classifiers = {
            'ABOD': ABOD(contamination=fixed_contamination),
            'KNN': KNN(contamination=fixed_contamination),
            'LOF': LOF(contamination=fixed_contamination),
            'One-Class SVM': OCSVM(contamination=fixed_contamination),
            'Isolation Forest': IForest(contamination=fixed_contamination, random_state=5)
        }

        # Evaluate each model
        for clf_name, clf in classifiers.items():
            clf.fit(X_train_scaled)
            y_pred = clf.predict(X_test_scaled)
            y_scores = clf.decision_function(X_test_scaled) if hasattr(clf, 'decision_function') else clf.predict_proba(X_test_scaled)[:, 1]

            result = {
                'Dataset Name': filename,
                'Total Missing Values': total_missing_values,
                'Number of Samples': X_train.shape[0] + X_test.shape[0],
                'Number of Features': X_train.shape[1],
                'Model': clf_name,
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_scores),
                'PR AUC': average_precision_score(y_test, y_scores),
                'MCC': matthews_corrcoef(y_test, y_pred),
                'Cohen\'s Kappa': cohen_kappa_score(y_test, y_pred)
            }
            results.append(result)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Save the combined results to a single CSV file
csv_file_path = os.path.join(folder_path, 'data_without_resample.csv')
results_df.to_csv(csv_file_path, index=False)
print(f'Combined results saved to {csv_file_path}')
