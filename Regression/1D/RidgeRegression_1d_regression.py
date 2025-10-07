import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from rdkit import Chem
import os

directory = "/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25"

file_list = [
    'V2_File1_Done.csv',
    'V2_File2_Done.csv',
    'V2_File3_Done.csv',
    'V2_File4_Done.csv',
    'V2_File5_Done.csv',
    'V2_File6_Done.csv',
    'V2_File7_Done.csv',
    'V2_File8_Done.csv',
    'V2_File9_Done.csv',
    'V2_File10_Done.csv',
    'V2_File11_Done.csv',
]

all_dataframes = []
for file_name in file_list:
    full_path = os.path.join(directory, file_name)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        all_dataframes.append(df)
    else:
        print(f"Warning: {file_name} not found")

if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)

df = combined_df.rename(columns={
    "Reaction ID": "Reaction_ID",
    "Yield": "Yield",
    "Temperature": "Temp",
    "Time": "Time",
    "Solvent SMILES": "Solvent_SMILES",
    "Solvent": "Solvent",
    "Coupling Agent SMILES": "CA_SMILES",
    "Coupling Agent": "CA",
    "COOH SMILES": "COOH_SMILES",
    "COOH": "COOH",
    "COOH MW": "COOH_MW",
    "COOH logP": "COOH_logP",
    "Amine SMILES": "Amine_SMILES",
    "Amine": "Amine",
    "Amine MW": "Amine_MW",
    "Amine logP": "Amine_logP",
    "Additive SMILES": "Additive_SMILES",
    "Additive": "Additive"
})

df.reset_index(inplace=True)
df.info()
df.shape

list_of_smiles = ["Solvent_SMILES", "CA_SMILES", "COOH_SMILES", "Amine_SMILES", "Additive_SMILES"]

def create_dummy_variables(df, categorical_cols):
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    return df

df = create_dummy_variables(df, list_of_smiles)

Solvent_SMILES_dummies = [col for col in df.columns if col.startswith('Solvent_SMILES_')]
CA_SMILES_dummies = [col for col in df.columns if col.startswith('CA_SMILES_')]
COOH_SMILES_dummies = [col for col in df.columns if col.startswith('COOH_SMILES_')]
Amine_SMILES_dummies = [col for col in df.columns if col.startswith('Amine_SMILES_')]
Additive_SMILES_dummies = [col for col in df.columns if col.startswith('Additive_SMILES_')]

print(len(Solvent_SMILES_dummies), len(CA_SMILES_dummies), len(COOH_SMILES_dummies),
      len(Amine_SMILES_dummies), len(Additive_SMILES_dummies))

df.head(5)

from sklearn.impute import SimpleImputer

num_features = ['Temp', 'Time', 'COOH_MW', 'Amine_MW', 'Amine_logP', 'COOH_logP']

imputer = SimpleImputer(strategy='median')
df.loc[:, num_features] = imputer.fit_transform(df[num_features])

train, test = train_test_split(df, test_size=0.2, random_state=1)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

features = num_features + CA_SMILES_dummies + Additive_SMILES_dummies + Solvent_SMILES_dummies + COOH_SMILES_dummies + Amine_SMILES_dummies

X_train = train[features]
y_train = train['Yield']
X_test = test[features]
y_test = test['Yield']

ridge_model = Ridge(alpha=10, random_state=42)
ridge_model.fit(X_train, y_train)

y_train_pred = ridge_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

print("Training Set Performance:")
print(f"Root Mean Squared Error: {rmse_train:.2f}")
print(f"R-squared Score: {r2_train:.3f}")

print("\n")

y_test_pred = ridge_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("Test Set Performance:")
print(f"Root Mean Squared Error: {rmse_test:.2f}")
print(f"R-squared Score: {r2_test:.3f}")
