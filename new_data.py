import numpy as np
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score
from pyod.utils.utility import standardizer

# Load your data
data = np.load(r'C:\Users\sakha\Downloads\ds\ds\Amusk.npz')

X_train = data['x']
X_test = data['tx']
y_test = data['ty']

# Checking and printing the original number of samples
original_samples_train = X_train.shape[0]
original_samples_test = X_test.shape[0]
print(f"Original Total Number of Training Samples: {original_samples_train}")
print(f"Original Total Number of Test Samples: {original_samples_test}")

# Remove missing values (if any)
X_train = X_train[~np.isnan(X_train).any(axis=1)]
X_test = X_test[~np.isnan(X_test).any(axis=1)]
y_test = y_test[~np.isnan(y_test)]

# Printing the number of samples after removing NaN values
samples_after_nan_removal_train = X_train.shape[0]
samples_after_nan_removal_test = X_test.shape[0]
print(f"Total Number of Training Samples after Removing NaN: {samples_after_nan_removal_train}")
print(f"Total Number of Test Samples after Removing NaN: {samples_after_nan_removal_test}")

# Standardize the data using PyOD's standardizer
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

# DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'F1 Score', 'ROC AUC', 'PR AUC', 'MCC', 'Cohen Kappa'])

# Iterate over classifiers and calculate metrics
for clf_name, clf in classifiers.items():
    clf.fit(X_train_scaled)
    y_pred = clf.predict(X_test_scaled)
    y_scores = clf.decision_function(X_test_scaled) if hasattr(clf, 'decision_function') else clf.predict_proba(X_test_scaled)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    results_df = results_df.append({
        'Model': clf_name,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'MCC': mcc,
        'Cohen Kappa': kappa
    }, ignore_index=True)

print(results_df)
