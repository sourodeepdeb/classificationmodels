import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
import os
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv('/content/drive/Shared drives/MLDD/ORD_Data/Main Data/train_04_13_25.csv')
test_df =  pd.read_csv('/content/drive/Shared drives/MLDD/ORD_Data/Main Data/test_04_13_25.csv')

train_df['data_type'] = 'Train'
test_df['data_type'] = 'Test'
combined_df = pd.concat([train_df, test_df], axis=0)
combined_df.head(2)

df = combined_df.rename(columns={
    "Reaction ID":"Reaction_ID",
    "Yield":"Yield",
    "Temperature":"Temp",
    "Time":"Time",
    "Solvent SMILES":"Solvent_SMILES",
    "Solvent":"Solvent",
    "Coupling Agent SMILES":"CA_SMILES",
    "Coupling Agent":"CA",
    "COOH SMILES":"COOH_SMILES",
    "COOH":"COOH",
    "COOH MW":"COOH_MW",
    "COOH logP":"COOH_logP",
    "Amine SMILES":"Amine_SMILES",
    "Amine":"Amine",
    "Amine MW":"Amine_MW",
    "Amine logP":"Amine_logP",
    "Additive SMILES":"Additive_SMILES",
    "Additive":"Additive"
})

df.reset_index(inplace=True)
df.info()

list_of_smiles = ["Solvent_SMILES", "CA_SMILES", "COOH_SMILES", "Amine_SMILES", "Additive_SMILES"]

def create_dummy_variables(df, categorical_cols):
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
    return df

df = create_dummy_variables(df, list_of_smiles)

Solvent_SMILES_dummies = [col for col in df.columns if col.startswith('Solvent_SMILES_')]
CA_SMILES_dummies = [col for col in df.columns if col.startswith('CA_SMILES_')]
COOH_SMILES_dummies = [col for col in df.columns if col.startswith('COOH_SMILES_')]
Amine_SMILES_dummies = [col for col in df.columns if col.startswith('Amine_SMILES_')]
Additive_SMILES_dummies = [col for col in df.columns if col.startswith('Additive_SMILES_')]

print(f"Number of dummy columns per group:")
print(f"Solvent: {len(Solvent_SMILES_dummies)} | CA: {len(CA_SMILES_dummies)} | COOH: {len(COOH_SMILES_dummies)} | Amine: {len(Amine_SMILES_dummies)} | Additive: {len(Additive_SMILES_dummies)}")
df.head(5)

num_features = ['Temp', 'Time', 'COOH_MW', 'Amine_MW', 'Amine_logP', 'COOH_logP']

features = num_features + CA_SMILES_dummies + Additive_SMILES_dummies + Solvent_SMILES_dummies + COOH_SMILES_dummies + Amine_SMILES_dummies

X_train = df[df['data_type'] == 'Train'][features]
y_train = df[df['data_type'] == 'Train']['Yield']
X_test = df[df['data_type'] == 'Test'][features]
y_test = df[df['data_type'] == 'Test']['Yield']

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2
)

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_train_pred)
print("Training Set Performance")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

print("\n\n")
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Test Set Performance")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

param_grid = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = random_search.best_estimator_
rf_model = best_model
rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_train_pred)
print("\nTraining Set Performance (Tuned Model)")
print(f"R-squared Score: {r2:.2f}")

print("\n\n")
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Test Set Performance (Tuned Model)")
print(f"R-squared Score: {r2:}")

feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features (Tuned Model):")
