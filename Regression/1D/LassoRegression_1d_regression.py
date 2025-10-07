import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

files = [
    f"/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File{i}_Done.csv"
    for i in range(1, 12)
]
df = pd.concat([pd.read_csv(file) for file in files], axis=0).reset_index(drop=True)

df = df.rename(columns={
    "Reaction ID": "Reaction_ID", "Yield": "Yield", "Temperature": "Temp", "Time": "Time",
    "Solvent SMILES": "Solvent_SMILES", "Solvent": "Solvent",
    "Coupling Agent SMILES": "CA_SMILES", "Coupling Agent": "CA",
    "COOH SMILES": "COOH_SMILES", "COOH": "COOH",
    "COOH MW": "COOH_MW", "COOH logP": "COOH_logP",
    "Amine SMILES": "Amine_SMILES", "Amine": "Amine",
    "Amine MW": "Amine_MW", "Amine logP": "Amine_logP",
    "Additive SMILES": "Additive_SMILES", "Additive": "Additive"
})

list_of_smiles = ["Solvent_SMILES", "COOH_SMILES", "Amine_SMILES", "Additive_SMILES", "CA_SMILES"]

def create_dummy_variables(df, categorical_cols):
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col].fillna(""), prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
    return df

df = create_dummy_variables(df, list_of_smiles)

Solvent_SMILES_dummies = [col for col in df.columns if col.startswith('Solvent_SMILES_')]
CA_SMILES_dummies = [col for col in df.columns if col.startswith('CA_SMILES_')]
COOH_SMILES_dummies = [col for col in df.columns if col.startswith('COOH_SMILES_')]
Amine_SMILES_dummies = [col for col in df.columns if col.startswith('Amine_SMILES_')]
Additive_SMILES_dummies = [col for col in df.columns if col.startswith('Additive_SMILES_')]

num_features = ['Temp', 'Time', 'COOH_MW', 'Amine_MW', 'Amine_logP', 'COOH_logP']
features = num_features + CA_SMILES_dummies + Additive_SMILES_dummies + Solvent_SMILES_dummies + COOH_SMILES_dummies + Amine_SMILES_dummies

df = df.dropna(subset=['Yield'])

X = df[features].fillna(0)
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_model = Lasso(alpha=0.16, random_state=42, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

print("Training Set Performance:")
y_train_pred = lasso_model.predict(X_train_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}")
print(f"R2: {r2_score(y_train, y_train_pred)}")

print("\nTest Set Performance:")
y_test_pred = lasso_model.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}")
print(f"R2: {r2_score(y_test, y_test_pred)}")

feature_importance = pd.DataFrame({'feature': features, 'importance': np.abs(lasso_model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

param_grid = { 'alpha': np.logspace(-4, 1, 100) }
random_search = RandomizedSearchCV(
    estimator=Lasso(random_state=42, max_iter=10000),
    param_distributions=param_grid,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train_scaled, y_train)

print("\nBest Hyperparameters:", random_search.best_params_)

best_lasso_model = random_search.best_estimator_

print("\nTuned Model Performance on Training Set:")
y_train_pred = best_lasso_model.predict(X_train_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}")
print(f"R2: {r2_score(y_train, y_train_pred)}")

print("\nTuned Model Performance on Test Set:")
y_test_pred = best_lasso_model.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}")
print(f"R2: {r2_score(y_test, y_test_pred)}")

feature_importance = pd.DataFrame({'feature': features, 'importance': np.abs(best_lasso_model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features (Tuned Model):")
print(feature_importance.head(10))
