import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_paths = [
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File1_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File2_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File3_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File4_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File5_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File6_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File7_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File8_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File9_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File10_Done.csv',
    '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File11_Done.csv'
]

df_list = [pd.read_csv(fp) for fp in file_paths]
df = pd.concat(df_list, axis=0).reset_index(drop=True)

df = df.rename(columns={
    "Reaction ID":"Reaction_ID", "Yield":"Yield", "Temperature":"Temp", "Time":"Time",
    "Solvent SMILES":"Solvent_SMILES", "Solvent":"Solvent",
    "Coupling Agent SMILES":"CA_SMILES", "Coupling Agent":"CA",
    "COOH SMILES":"COOH_SMILES", "COOH":"COOH",
    "COOH MW":"COOH_MW", "COOH logP":"COOH_logP",
    "Amine SMILES":"Amine_SMILES", "Amine":"Amine",
    "Amine MW":"Amine_MW", "Amine logP":"Amine_logP",
    "Additive SMILES":"Additive_SMILES", "Additive":"Additive"
})

list_of_smiles = ["Solvent_SMILES", "COOH_SMILES", "Amine_SMILES", "Additive_SMILES", "CA_SMILES"]

def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

for col in list_of_smiles:
    print(f"Processing SMILES for {col}...")
    fp_array = np.array([smiles_to_fingerprint(sm) for sm in df[col].fillna('')])
    fp_df = pd.DataFrame(fp_array, columns=[f"{col}_fp_{i}" for i in range(fp_array.shape[1])])
    df = pd.concat([df, fp_df], axis=1)

fp_features = [col for col in df.columns if "fp_" in col]

df['Sum_CA_fp'] = df[[col for col in df.columns if "CA_SMILES_fp_" in col]].sum(axis=1)
df['Sum_COOH_fp'] = df[[col for col in df.columns if "COOH_SMILES_fp_" in col]].sum(axis=1)
df['Sum_Amine_fp'] = df[[col for col in df.columns if "Amine_SMILES_fp_" in col]].sum(axis=1)
df['Sum_Additive_fp'] = df[[col for col in df.columns if "Additive_SMILES_fp_" in col]].sum(axis=1)
df['Sum_Solvent_fp'] = df[[col for col in df.columns if "Solvent_SMILES_fp_" in col]].sum(axis=1)

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

num_features = ['Temp', 'Time', 'COOH_MW', 'Amine_MW', 'Amine_logP', 'COOH_logP']
features = num_features + fp_features + CA_SMILES_dummies + Additive_SMILES_dummies + Solvent_SMILES_dummies + COOH_SMILES_dummies + Amine_SMILES_dummies

df = df.dropna(subset=['Yield'])
X = df[features].fillna(0)
y = df['Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

en_model = ElasticNet(alpha=0.12, l1_ratio=0.5, random_state=42, max_iter=10000)
en_model.fit(X_train_scaled, y_train)

print("Initial Model Performance:")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, en_model.predict(X_train_scaled))):}")
print(f"Train R2: {r2_score(y_train, en_model.predict(X_train_scaled)):}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, en_model.predict(X_test_scaled))):}")
print(f"Test R2: {r2_score(y_test, en_model.predict(X_test_scaled)):}")

feature_importance = pd.DataFrame({'feature': features, 'importance': np.abs(en_model.coef_)})
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values('importance', ascending=False).head(10))

param_grid = {
    'alpha': np.logspace(-4, 1, 100),
    'l1_ratio': np.linspace(0.1, 1.0, 10)
}

random_search = RandomizedSearchCV(
    estimator=ElasticNet(random_state=42, max_iter=10000),
    param_distributions=param_grid,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)

best_en_model = random_search.best_estimator_
print("\nBest Hyperparameters:", random_search.best_params_)

y_train_pred = best_en_model.predict(X_train_scaled)
y_test_pred = best_en_model.predict(X_test_scaled)

print("\nBest Model Performance:")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):}")
print(f"Train R2: {r2_score(y_train, y_train_pred):}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):}")
print(f"Test R2: {r2_score(y_test, y_test_pred):}")

feature_importance = pd.DataFrame({'feature': features, 'importance': np.abs(best_en_model.coef_)})
print("\nTop 10 Most Important Features (Tuned Model):")
print(feature_importance.sort_values('importance', ascending=False).head(10))
