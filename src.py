#####################################
# STEP 1. Import necessary libraries
#####################################
import pandas as pd                                     
import numpy as  np           
import pywt                                                     
from sklearn.preprocessing import StandardScaler                    
from sklearn.model_selection import train_test_split     
import matplotlib.pyplot as plt                          
import seaborn as sns    
from imblearn.over_sampling import SMOTE               
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay            
from imblearn.over_sampling import SMOTE                 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score     
from sklearn.decomposition import PCA                                        
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
import joblib
import pickle
import os

######################################################
# STEP 2. Loading the dataset
######################################################
data = pd.read_csv("Signal_Data.csv")
data.head(5)

######################################################
# STEP 3. Pre-processing
######################################################
print("Dataset Information:")
print(data.info())  

# Handling Missing and null values
missing_values = data.isnull().sum().sum()
print("\nMissing Values in dataset:", missing_values)

print("Dimensions of Dataset:")
print(data.shape)

print("Data summary statistics: \n")
print(data.describe())

missing_values = data.isnull().sum().sum()
print("\nMissing Values in dataset:", missing_values)


null_values = data.isna().sum().sum()
print("\nNull Values in dataset:", null_values)
null_values = data.isna().sum().sum()
print("\nNull Values in dataset:", null_values)

# Data splitting into input (X) and target (y)
X = data.drop(columns=['column_a', 'y'])  
y = data['y']

# Combine non-epileptic classes (2, 3, 4, 5) into a single class (0)
y_combined = y.copy()
y_combined[y_combined != 1] = 0  # Class 1 is epileptic, class 0 is Non-epileptic

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)

######################################################
# STEP 4. Oversampling - Balancing the dataset
######################################################
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f'Original dataset shape: {X_train.shape}')
print(f'Resampled dataset shape: {X_train_resampled.shape}')
print(f'Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts()}')

######################################################
# STEP 5. Feature scaling 
######################################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
print(f'Scaled Resampled Data Shape: {X_train_scaled.shape}')
print(f'Test Data Shape: {X_test_scaled.shape}')

######################################################
# STEP 6. Data Transformation/Feature Extraction -DWT
######################################################
def extract_dwt_features(data):
    features = []
    for signal in data:
        coeffs = pywt.wavedec(signal, wavelet='db4', level=5)
        flattened_coeffs = np.hstack(coeffs)  
        features.append(flattened_coeffs)
    return np.array(features)

# Extract DWT features from the scaled training and test data
X_train_dwt = extract_dwt_features(X_train_scaled)
X_test_dwt = extract_dwt_features(X_test_scaled)
print(f'DWT Feature Shape for Resampled Data: {X_train_dwt.shape}')
print(f'DWT Feature Shape for Test Data: {X_test_dwt.shape}')

######################################################
# STEP 7. Dimensionality reduction - PCA
######################################################
pca = PCA(n_components=0.95)  # Retain components that explain 95% of the variance
X_train_pca = pca.fit_transform(X_train_dwt)
X_test_pca = pca.transform(X_test_dwt)
print(f'PCA Transformed Shape for Resampled Data: {X_train_pca.shape}')
print(f'PCA Transformed Shape for Test Data: {X_test_pca.shape}')


######################################################
# STEP 8. Model Training
######################################################
classifiers = {
    'XGBoost': XGBClassifier(eval_metric='mlogloss'),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'MLP': MLPClassifier(max_iter=500)
}

# Train each classifier and display performance metrics
for name, clf in classifiers.items():
    clf.fit(X_train_pca, y_train_resampled) 
    y_pred = clf.predict(X_test_pca)  

    # Calculate and display evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Display confusion matrix and metrics
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nClassifier: {name}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

###################################################################
# STEP 9. Soft Voting Classifier (Ensemble of multiple classifiers)
###################################################################
clf1 = XGBClassifier(eval_metric='mlogloss')
clf2 = KNeighborsClassifier()
clf3 = DecisionTreeClassifier()
clf4 = RandomForestClassifier()
clf5 = MLPClassifier(max_iter=500)

# Create a voting classifier with soft voting
voting_clf = VotingClassifier(estimators=[
    ('XGBoost', clf1), 
    ('KNN', clf2), 
    ('Decision Tree', clf3), 
    ('Random Forest', clf4), 
    ('MLP', clf5)], 
    voting='soft')

# Train the voting classifier
voting_clf.fit(X_train_pca, y_train_resampled)

# Predict on the test data
y_pred = voting_clf.predict(X_test_pca)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

####################################################################
# STEP 10. Performance metrics for the voting classifier
####################################################################

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix for percentage representation
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent,
            annot=True,
            fmt='.2f',
            cmap='Oranges',
            xticklabels=['Non-Epileptic', 'Epileptic'],
            yticklabels=['Non-Epileptic', 'Epileptic'],
            linewidths=1,
            linecolor='black',
            annot_kws={"size": 16})  # Corrected placement of annot_kws

# Add titles and labels
plt.title('Confusion Matrix for Epileptic Predictive System', fontsize=16, pad=20)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)

# Customize tick parameters
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show(block=False)

# Display results
print(f"\nSoft Voting Classifier Results:")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred))

####################################################################
# STEP 11. Cross-validation using K-fold cross-validation
####################################################################

# Define a custom scoring metric (AUC)
auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

# Perform k-fold cross-validation 
k = 5  # Number of folds
cv_scores = cross_val_score(voting_clf, X_train_pca, y_train_resampled, cv=k, scoring=auc_scorer)

# Display cross-validation results
print(f"Cross-Validation AUC Scores: {cv_scores}")
print(f"Mean AUC: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of AUC: {np.std(cv_scores):.4f}")

####################################################################
# STEP 12. Receiver operating characteristic (ROC) curve
####################################################################

# Get predicted probabilities for the positive class (epileptic: 1)
y_pred_proba = voting_clf.predict_proba(X_test_pca)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show(block=False)

####################################################################
# STEP 13. Save the trained model
####################################################################

# Save the trained model to a file using pickle
model_filename = 'Model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(voting_clf, model_file) 

# Print model save confirmation with absolute path
print("Model saved successfully!")
print("Model file path:", model_filename)
