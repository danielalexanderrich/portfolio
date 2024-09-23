import pandas as pd
import os
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
import matplotlib

os.chdir('C:\\Users\\Dan\\OneDrive\\Desktop\\data sets')
creditData = pd.read_csv('loan.csv')


##need to make a new target where each class in the target variable is assigned to a number due to how lightgbm handles types
targetClasses = creditData['loan_status'].unique()
targetNumerical = {targetClasses[x]: x for x in range(len(targetClasses))}
creditData['loan_status_class'] = creditData['loan_status'].map(targetNumerical)

##explore data and organize by type
notesFrame = pd.DataFrame(creditData.columns.tolist())
notesFrame = notesFrame.rename(columns={0: "feature"})
types = []
for column in creditData:
    types.append(creditData[column].dtype)
notesFrame['type'] = types
notesFrame = notesFrame.sort_values(by='type')

##get just the numerical features, will work on rest later
numericalFeatures = notesFrame[notesFrame['type'] != 'object']
numericalFeatures = numericalFeatures['feature'].tolist()
numericalData = creditData.copy()
numericalData = numericalData[numericalFeatures]
#numericalData['target'] = creditData['loan_status_class']

###build model on numerical features

# Define features and target
numericalData = numericalData.drop(['id', 'member_id'], axis=1)
X = numericalData.drop('loan_status_class', axis=1)
y = numericalData['loan_status_class']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


params = {
    'objective': 'multiclass',  # for multi-class classification
    'metric': 'multi_logloss',
    'num_class': len(targetClasses),
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}
# Train the model
evals = [train_data, val_data]
bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=evals)

# Make predictions
y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred_classes = y_pred.argmax(axis=1)

# Evaluate
print("Accuracy:", accuracy_score(y_val, y_pred_classes))
##Accuracy: 0.9682717663233339??? wayyyyy too high definitely overfit


params = {
    'objective': 'multiclass',  # for multi-class classification
    'metric': 'multi_logloss',
    'num_class': len(targetClasses),
    'boosting_type': 'gbdt',
    'min_data_in_leaf': 20, #hopefully limit accuracy
    'max_depth': 5, #hopefully limit accuracy
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}
# Train the model
evals = [train_data, val_data]
bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=evals)

# eval new preds
y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred_classes = y_pred.argmax(axis=1)
print("Accuracy:", accuracy_score(y_val, y_pred_classes))
##Accuracy: 0.9679618652662896 lower but not low enough

#### is there a difference between regressor above and classifier?
model = lgb.LGBMClassifier(objective='multiclass', num_class=len(y.unique()), metric='multi_logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_val, num_iteration=bst.best_iteration)
print("Accuracy:", accuracy_score(y_val, y_pred))


importances = model.feature_importances_
# Create a DataFrame to hold feature names and their importances
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
})
# Sort the DataFrame by Importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


sample_size = 1000
sample_indices = np.random.choice(X_val.index, sample_size, replace=False)
X_val_sample = X_val.loc[sample_indices]
# Specify the features you want to analyze (can use feature names or indices)
features = [0, 1]  # Example feature indices

# Create partial dependence plots for each class
# Plot for all classes
fig, ax = plt.subplots(figsize=(12, 8))
display = PartialDependenceDisplay.from_estimator(model, X_val, features, target=0, ax=ax, grid_resolution=50) ###taking a long time on full data, want to subset probably

plt.show()
