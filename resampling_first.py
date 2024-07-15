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

folder_path = r'C:\Users\sakha\Downloads\ds\all dataset'   # Replace with the path to your datasets folder
file_extension = '.npz'
results = []


def resample_test_set(X_test, y_test, anomaly_percentage):
    normal_data = X_test[y_test == 0]
    anomaly_data = X_test[y_test == 1]

    n_anomalies = int(len(normal_data) * anomaly_percentage / (1 - anomaly_percentage))
    n_anomalies = min(n_anomalies, len(anomaly_data))

    resampled_anomalies = anomaly_data[:n_anomalies]
    resampled_test = np.vstack([normal_data, resampled_anomalies])
    resampled_labels = np.array([0] * len(normal_data) + [1] * n_anomalies)

    return resampled_test, resampled_labels


# Loop over each file in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(file_extension):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)

        X_train = data['x']
        X_test = data['tx']
        y_test = data['ty']

        # Remove missing values
        X_train = X_train[~np.isnan(X_train).any(axis=1)]
        X_test = X_test[~np.isnan(X_test).any(axis=1)]
        y_test = y_test[~np.isnan(y_test)]

        # Initial anomaly percentage in the test set
        initial_anomaly_percentage = np.mean(y_test) * 100

        # Resample test set to have approx 10% anomalies
        X_test_resampled, y_test_resampled = resample_test_set(X_test, y_test, 0.1)
        resampled_anomaly_percentage = np.mean(y_test_resampled) * 100

        # Standardize the data
        X_train_scaled, X_test_scaled = standardizer(X_train, X_test_resampled)

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
            y_scores = clf.decision_function(X_test_scaled) if hasattr(clf, 'decision_function') else clf.predict_proba(
                X_test_scaled)[:, 1]

            result = {
                'Dataset Name': filename,
                'Initial Anomaly Percentage': initial_anomaly_percentage,
                'Resampled Anomaly Percentage': resampled_anomaly_percentage,
                'Number of Features': X_train.shape[1],
                'Model': clf_name,
                'F1 Score': f1_score(y_test_resampled, y_pred),
                'ROC AUC': roc_auc_score(y_test_resampled, y_scores),
                'PR AUC': average_precision_score(y_test_resampled, y_scores),
                'MCC': matthews_corrcoef(y_test_resampled, y_pred),
                'Cohen\'s Kappa': cohen_kappa_score(y_test_resampled, y_pred)
            }
            results.append(result)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Save the combined results to a single CSV file
csv_file_path = os.path.join(folder_path, 'resampled_results.csv')
results_df.to_csv(csv_file_path, index=False)
print(f'Resampled results saved to {csv_file_path}')
