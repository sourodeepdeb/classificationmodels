import os
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

files = [f"/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File{i}_Done.csv" for i in range(1, 11+1)]
data_list = []
for file in files:
    if os.path.exists(file):
        data_list.append(pd.read_csv(file))

uploaded_path = "/mnt/data/V2_File1_Done_with_optXYZ.csv"
if os.path.exists(uploaded_path):
    data_list.append(pd.read_csv(uploaded_path))

if not data_list:
    raise FileNotFoundError("No input CSVs found. Check your paths.")

combined_data = pd.concat(data_list, ignore_index=True)

numerical_features = ["Time", "Temperature", "COOH MW", "COOH logP", "Amine MW", "Amine logP"]
smiles_columns = ["Solvent SMILES", "Coupling Agent SMILES", "COOH SMILES", "Amine SMILES", "Additive SMILES"]
target = "Yield"

combined_data[numerical_features] = combined_data[numerical_features].apply(pd.to_numeric, errors='coerce')
combined_data[numerical_features] = combined_data[numerical_features].fillna(combined_data[numerical_features].median())

for col in smiles_columns:
    if col not in combined_data.columns:
        combined_data[col] = ""
    combined_data[col] = combined_data[col].fillna("").astype(str)

if target not in combined_data.columns:
    raise KeyError(f"Target column '{target}' not found.")
combined_data[target] = pd.to_numeric(combined_data[target], errors='coerce').clip(0, 100)

def parse_xyz(xyz_str):
    if not isinstance(xyz_str, str) or not xyz_str.strip():
        return [], None
    elems = []
    coords = []
    for line in xyz_str.strip().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            el = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except Exception:
                continue
            elems.append(el)
            coords.append([x, y, z])
    if len(coords) < 3:
        return [], None
    return elems, np.asarray(coords, dtype=float)

def three_d_features(elems, coords, prefix):
    """Compute 3D shape features + pairwise distances for one molecule."""
    out = {}
    if coords is None or len(coords) < 3:
        for k in [
            'radiusGyration', 'PMI1', 'PMI2', 'PMI3',
            'numAtoms', 'pair_min', 'pair_mean', 'pair_max', 'pair_std',
            'nn_mean', 'nn_std'
        ]:
            out[f"{prefix}_{k}"] = np.nan
        return out

    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    rg = float(np.sqrt(np.mean(dists**2)))

    cov = np.cov(coords.T)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        pmi1, pmi2, pmi3 = map(float, np.sort(eigvals))
    except Exception:
        pmi1 = pmi2 = pmi3 = np.nan

    from scipy.spatial.distance import pdist, squareform
    if len(coords) >= 2:
        pd = pdist(coords)
        pair_min = float(np.min(pd)) if pd.size else np.nan
        pair_max = float(np.max(pd)) if pd.size else np.nan
        pair_mean = float(np.mean(pd)) if pd.size else np.nan
        pair_std = float(np.std(pd)) if pd.size else np.nan

        dm = squareform(pd) if pd.size else np.zeros((len(coords), len(coords)))
        nn = np.partition(dm + np.eye(len(coords))*1e9, 1, axis=1)[:, 1]
        nn_mean = float(nn.mean())
        nn_std = float(nn.std())
    else:
        pair_min = pair_mean = pair_max = pair_std = np.nan
        nn_mean = nn_std = np.nan

    out.update({
        f"{prefix}_radiusGyration": rg,
        f"{prefix}_PMI1": pmi1,
        f"{prefix}_PMI2": pmi2,
        f"{prefix}_PMI3": pmi3,
        f"{prefix}_numAtoms": int(len(coords)),
        f"{prefix}_pair_min": pair_min,
        f"{prefix}_pair_mean": pair_mean,
        f"{prefix}_pair_max": pair_max,
        f"{prefix}_pair_std": pair_std,
        f"{prefix}_nn_mean": nn_mean,
        f"{prefix}_nn_std": nn_std,
    })
    return out

def try_soap(elems, coords, prefix):
    try:
        from ase import Atoms
        from dscribe.descriptors import SOAP
    except Exception:
        return {} 

    if coords is None or len(coords) < 1:
        return {}

    try:
        atoms = Atoms(symbols=elems, positions=coords)
    except Exception:
        return {}

    species = sorted(set(elems))
    soap = SOAP(
        species=species,
        rcut=3.0,
        nmax=2,
        lmax=2,
        sigma=0.5,
        periodic=False,
        sparse=False
    )
    vec = soap.create(atoms, positions=[i for i in range(len(atoms))])
    soap_mean = np.asarray(vec).mean(axis=0)
    return {f"{prefix}_SOAP_{i}": float(v) for i, v in enumerate(soap_mean)}

xyz_map = {
    "COOH": "COOH optXYZ",
    "Amine": "Amine optXYZ" if "Amine optXYZ" in combined_data.columns else ("AMINE optXYZ" if "AMINE optXYZ" in combined_data.columns else None),
}

all_desc_frames = []
for prefix, colname in xyz_map.items():
    if colname is None or colname not in combined_data.columns:
        continue

    elems_coords = combined_data[colname].apply(parse_xyz)
    elems_list = [ec[0] for ec in elems_coords]
    coords_list = [ec[1] for ec in elems_coords]

    rows = []
    for elems, coords in zip(elems_list, coords_list):
        row = {}
        row.update(three_d_features(elems, coords, prefix))
        row.update(try_soap(elems, coords, prefix))
        rows.append(row)

    desc_df = pd.DataFrame(rows)
    all_desc_frames.append(desc_df)

if all_desc_frames:
    desc_all = pd.concat(all_desc_frames, axis=1)
    combined_data = pd.concat([combined_data, desc_all], axis=1)
    new_num_cols = [c for c in desc_all.columns if desc_all[c].dtype != 'O']
    numerical_features.extend(new_num_cols)

combined_data[numerical_features] = combined_data[numerical_features].apply(pd.to_numeric, errors='coerce')
combined_data[numerical_features] = combined_data[numerical_features].fillna(combined_data[numerical_features].median())


class MorganFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_cols, radius=2, n_bits=2048):
        self.smiles_cols = smiles_cols
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X, y=None):
        return self

    def _smiles_to_bits(self, s):
        if not isinstance(s, str) or not s:
            arr = np.zeros(self.n_bits, dtype=np.uint8)
            return arr
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            arr = np.zeros(self.n_bits, dtype=np.uint8)
            return arr
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    def transform(self, X):
        mats = []
        for col in self.smiles_cols:
            vals = X[col].values
            rows = [self._smiles_to_bits(s) for s in vals]
            mat = sparse.csr_matrix(np.vstack(rows))
            mats.append(mat)
        return sparse.hstack(mats, format='csr')

feature_cols = list(set(numerical_features + smiles_columns))  
X = combined_data[feature_cols]
y = combined_data[target].astype(float)

smiles_tfidf_steps = [
    (col, TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)), col) for col in smiles_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [c for c in numerical_features if c in X.columns]),
        *smiles_tfidf_steps,
        ('morgan', MorganFeaturizer(smiles_cols=smiles_columns, radius=2, n_bits=1024), smiles_columns),
    ],
    remainder='drop',
    sparse_threshold=0.3  
)

model = MLPRegressor(
    random_state=42,
    max_iter=600,         
    early_stopping=True,  
    n_iter_no_change=20,
    validation_fraction=0.1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

param_dist = {
    'model__hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128)],
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam'],
    'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    'model__learning_rate': ['constant', 'adaptive'],
    'model__learning_rate_init': [1e-3, 5e-4, 1e-4],
    'model__beta_1': [0.9, 0.95],
    'model__beta_2': [0.999, 0.99],
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,          
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='r2',
    random_state=1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nBest Parameters:", random_search.best_params_)
print(f"r^2 Score: {r2}")
