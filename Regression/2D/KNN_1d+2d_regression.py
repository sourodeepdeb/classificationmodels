import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

data_paths = [
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

all_data = pd.DataFrame()
for path in data_paths:
    try:
        df = pd.read_csv(path)
        if 'Reaction ID' in df.columns:
            df = df.drop(columns=['Reaction ID'])
        all_data = pd.concat([all_data, df], ignore_index=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")

data = all_data.copy()
data = data[data['Yield'] < 100]
data = data[data['Yield'] != 0]
data = data.replace('None', np.nan)
data = data.dropna(subset=['Yield'])

def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

rdkit_descriptors = [
    'MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
    'TPSA', 'HeavyAtomCount', 'NumAromaticRings', 'FractionCSP3', 'RingCount',
    'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    'NumAliphaticRings', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
    'NumSaturatedRings', 'NumHeteroatoms'
]

def compute_rdkit_descriptors(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return [np.nan] * len(rdkit_descriptors)
    return [getattr(Descriptors, desc)(mol) for desc in rdkit_descriptors]

def compute_morgan_fingerprints(smiles, radius=2, n_bits=1024):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return [0] * n_bits
    fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return list(fp.ToBitString())

smiles_roles = [
    ('COOH SMILES', 'COOH'),
    ('Amine SMILES', 'Amine'),
    ('Additive SMILES', 'Additive'),
    ('Coupling Agent SMILES', 'CouplingAgent'),
    ('Solvent SMILES', 'Solvent')
]

for smiles_col, prefix in smiles_roles:
    if smiles_col in data.columns:
        desc_data = data[smiles_col].apply(compute_rdkit_descriptors).apply(pd.Series)
        desc_data.columns = [f'{prefix}_{desc}' for desc in rdkit_descriptors]
        data = pd.concat([data, desc_data], axis=1)

        fp_data = data[smiles_col].apply(lambda x: compute_morgan_fingerprints(x)).apply(pd.Series)
        fp_data.columns = [f'{prefix}_FP_{i}' for i in range(fp_data.shape[1])]
        data = pd.concat([data, fp_data], axis=1)

descriptor_cols = [col for col in data.columns if any(x in col for x in ['_MolWt', '_MolLogP', '_NumH', '_NumRot', '_TPSA',
                                                                      '_HeavyAtom', '_NumArom', '_Fraction', '_RingCount',
                                                                      '_NHOHCount', '_NOCount', '_NumAliphatic', '_NumSaturated',
                                                                      '_NumHeteroatoms'])]
fp_cols = [col for col in data.columns if '_FP_' in col]

X = data[descriptor_cols + fp_cols]
y = data['Yield']

imputer_desc = SimpleImputer(strategy='median')
X_desc = imputer_desc.fit_transform(X[descriptor_cols])

X_fp = X[fp_cols].values

X_combined = np.hstack([X_desc, X_fp])

scaler = StandardScaler()
X_combined[:, :len(descriptor_cols)] = scaler.fit_transform(X_combined[:, :len(descriptor_cols)])

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=3,
    subsample=0.8,
    max_features=0.5,
    random_state=42,
    verbose=1
)

gbr.fit(X_train, y_train)

y_pred_test = gbr.predict(X_test)

test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Test R²: {test_r2:.3f}")
print(f"Test MAE: {test_mae:.3f}")

plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred_test, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Yield (Test)")
plt.ylabel("Predicted Yield (Test)")
plt.title("Test Set: Actual vs Predicted Yield (Descriptors + Fingerprints)")
plt.grid(True)
plt.tight_layout()
plt.show()

feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)[-20:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(descriptor_cols + fp_cols)[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.show()
