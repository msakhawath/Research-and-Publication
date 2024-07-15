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

folder_path = r'C:\Users\sakha\Downloads\ds\all dataset'  # Replace with your folder path
file_extension = '.npz'
results = []
skipped_datasets = []
anomaly_percentages = [0.1, 0.2, 0.3, 0.4]

def remove_nan_values(X):
    return X[~np.isnan(X).any(axis=1)]

def adjust_test_set(X_train, X_test, y_test, anomaly_percentage):
    normal_data = X_train
    anomaly_data = X_test[y_test == 1]

    num_anomalies = int(len(anomaly_data))
    num_normals = int(num_anomalies * (1 - anomaly_percentage) / anomaly_percentage)

    if num_normals > len(normal_data):
        return None, None, None

    normal_data_test = normal_data[:num_normals]
    X_test_adjusted = np.vstack([normal_data_test, anomaly_data])
    y_test_adjusted = np.array([0] * num_normals + [1] * num_anomalies)
    X_train_adjusted = normal_data[num_normals:]

    return X_train_adjusted, X_test_adjusted, y_test_adjusted

def is_any_nan(*arrays):
    return any(np.isnan(arr).any() for arr in arrays)

for filename in os.listdir(folder_path):
    if filename.endswith(file_extension):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)

        X_train_original = remove_nan_values(data['x'])
        X_test_original = remove_nan_values(data['tx'])
        y_test_original = data['ty'][~np.isnan(data['ty'])]

        for anomaly_percentage in anomaly_percentages:
            X_train = np.copy(X_train_original)
            X_test = np.copy(X_test_original)
            y_test = np.copy(y_test_original)

            initial_anomaly_percentage = np.mean(y_test) * 100
            X_train_adjusted, X_test_adjusted, y_test_adjusted = adjust_test_set(X_train, X_test, y_test, anomaly_percentage)

            if X_train_adjusted is None:
                skipped_datasets.append((filename, anomaly_percentage))
                continue

            if is_any_nan(X_train_adjusted, X_test_adjusted, y_test_adjusted):
                skipped_datasets.append((filename, anomaly_percentage))
                continue

            resampled_anomaly_percentage = np.mean(y_test_adjusted) * 100
            X_train_scaled, X_test_scaled = standardizer(X_train_adjusted, X_test_adjusted)

            classifiers = {
                'ABOD': ABOD(contamination=0.1),
                'KNN': KNN(contamination=0.1),
                'LOF': LOF(contamination=0.1),
                'One-Class SVM': OCSVM(contamination=0.1),
                'Isolation Forest': IForest(contamination=0.1, random_state=5)
            }

            for clf_name, clf in classifiers.items():
                clf.fit(X_train_scaled)
                y_pred = clf.predict(X_test_scaled)
                y_scores = clf.decision_function(X_test_scaled) if hasattr(clf, 'decision_function') else clf.predict_proba(X_test_scaled)[:, 1]

                if is_any_nan(y_pred, y_scores):
                    skipped_datasets.append((filename, anomaly_percentage))
                    continue

                result = {
                    'Dataset Name': filename,
                    'Anomaly Percentage': anomaly_percentage * 100,
                    'Initial Anomaly Percentage': initial_anomaly_percentage,
                    'Resampled Anomaly Percentage': resampled_anomaly_percentage,
                    'Number of Features': X_train_adjusted.shape[1],
                    'Model': clf_name,
                    'F1 Score': f1_score(y_test_adjusted, y_pred),
                    'ROC AUC': roc_auc_score(y_test_adjusted, y_scores),
                    'PR AUC': average_precision_score(y_test_adjusted, y_scores),
                    'MCC': matthews_corrcoef(y_test_adjusted, y_pred),
                    'Cohen\'s Kappa': cohen_kappa_score(y_test_adjusted, y_pred)
                }
                results.append(result)

results_df = pd.DataFrame(results)
csv_file_path = os.path.join(folder_path, 'multiple_percentages_all.csv')
results_df.to_csv(csv_file_path, index=False)

print(f'Adjusted results saved to {csv_file_path}')
if skipped_datasets:
    print("Skipped datasets due to issues (insufficient normal samples, NaN values, etc.):")
    for dataset, percentage in skipped_datasets:
        print(f"  - Dataset: {dataset}, Anomaly Percentage: {percentage * 100}%")
