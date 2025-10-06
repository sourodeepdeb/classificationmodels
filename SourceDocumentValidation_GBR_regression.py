import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from dscribe.descriptors import SOAP
from ase import Atoms

pathToDir = "/content/drive/MyDrive/ordMLFiles"
allData = pd.DataFrame()
for filename in os.listdir(pathToDir):
    if filename.endswith(".csv") and "MedChem_AMIDE_with_optXYZ" not in filename and "optXYZ.csv" not in filename:
        pathToFile = os.path.join(pathToDir, filename)
        data = pd.read_csv(pathToFile)
        if "Reaction ID" in data.columns:
            data = data.drop(columns=["Reaction ID"])
        allData = pd.concat([allData, data], ignore_index=True)
data = allData.copy()
data = data[data['Yield'] < 100]
data = data[data['Yield'] != 0]
data = data.replace('None', np.nan)
data = data.dropna(subset=['Yield'])
if 'Temperature' in data.columns and 'Time' in data.columns:
    data['TempTimeInteraction'] = data['Temperature'] * data['Time']

def molFromSmiles(smiles):
    try: return Chem.MolFromSmiles(smiles)
    except: return None

def morgan_fp(smiles, n_bits=128):
    mol = molFromSmiles(smiles)
    if mol is None: return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def parse_xyz_string(xyz_string):
    try:
        if pd.isna(xyz_string) or not isinstance(xyz_string, str): return None
        lines = xyz_string.strip().split('\n')
        coords = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 4:
                coords.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        if len(coords) < 2: return None
        return np.array(coords)
    except Exception: return None

def compute_3d_descriptors(coords):
    try:
        if coords is None or len(coords) < 2: return [np.nan]*5
        centroid = coords.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
        pairwise_distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        mean_dist = pairwise_distances[np.triu_indices(len(coords), 1)].mean()
        std_dist = pairwise_distances[np.triu_indices(len(coords), 1)].std()
        inertia = np.cov(coords.T)
        eigvals = np.linalg.eigvalsh(inertia)
        eigvals_sorted = np.sort(eigvals)[::-1]
        if len(eigvals_sorted) < 3:
            eigvals_sorted = np.pad(eigvals_sorted, (0, 3-len(eigvals_sorted)), constant_values=np.nan)
        return [rg, mean_dist, std_dist, eigvals_sorted[0], eigvals_sorted[1]]
    except Exception: return [np.nan]*5

def mol_to_ase(mol):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        return Atoms(symbols=symbols, positions=positions)
    except Exception: return None

descNames = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
smilesRoles = [
    ('COOH SMILES', 'COOH'),
    ('Amine SMILES', 'Amine'),
    ('Additive SMILES', 'Additive'),
    ('Coupling Agent SMILES', 'Coupling Agent'),
    ('Solvent SMILES', 'Solvent')
]
fp_bits = 128
numericalFeatures = [
    'Temperature', 'Time', 'TempTimeInteraction',
    'COOH MW', 'COOH logP', 'Amine MW', 'Amine logP'
]
categoricalFeatures = [
    'Solvent', 'Coupling Agent', 'COOH', 'Amine', 'Additive', "Amine SMILES", "COOH SMILES"
]

smiles_columns = []
for col, _ in smilesRoles:
    if col in data.columns:
        smiles_columns.append(col)
elements = set(["H", "C", "N", "O"])
for col in smiles_columns:
    for smi in data[col].dropna().unique():
        mol = molFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                elements.add(atom.GetSymbol())
all_species = sorted(elements)
soap = SOAP(species=all_species, periodic=False, r_cut=5.0, n_max=4, l_max=2, sigma=0.3)
def get_soap(smiles):
    try:
        mol = molFromSmiles(smiles)
        if mol is None: return np.zeros(soap.get_number_of_features())
        atoms = mol_to_ase(mol)
        if atoms is None: return np.zeros(soap.get_number_of_features())
        soap_vec = soap.create(atoms)
        return soap_vec.mean(axis=0)
    except Exception: return np.zeros(soap.get_number_of_features())

for smilesCol, prefix in smilesRoles:
    if smilesCol in data.columns:
        molCol = f'{prefix}_Mol'
        data[molCol] = data[smilesCol].apply(molFromSmiles)
        descDf = data[molCol].apply(lambda m: [
            Descriptors.MolWt(m) if m else np.nan,
            Descriptors.MolLogP(m) if m else np.nan,
            Descriptors.TPSA(m) if m else np.nan,
            Descriptors.NumHDonors(m) if m else np.nan,
            Descriptors.NumHAcceptors(m) if m else np.nan,
            Descriptors.NumRotatableBonds(m) if m else np.nan
        ]).apply(pd.Series)
        descDf.columns = [f'{prefix}_{n}' for n in descNames]
        data = pd.concat([data, descDf], axis=1)
        data = data.drop(columns=[molCol])
        for n in descNames:
            col = f'{prefix}_{n}'
            if col in data.columns and col not in numericalFeatures:
                numericalFeatures.append(col)
        fps = data[smilesCol].fillna('').apply(lambda s: morgan_fp(s, n_bits=fp_bits))
        fp_df = pd.DataFrame(fps.tolist(), columns=[f'{prefix}_FP_{i}' for i in range(fp_bits)], index=data.index)
        data = pd.concat([data, fp_df], axis=1)
        for i in range(fp_bits):
            col = f'{prefix}_FP_{i}'
            if col in data.columns and col not in numericalFeatures:
                numericalFeatures.append(col)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), min_df=1, max_features=32)
        smiles_vec = vectorizer.fit_transform(data[smilesCol].fillna('')).toarray()
        smiles_vec_names = [f"{prefix}_SMILES_{f}" for f in vectorizer.get_feature_names_out()]
        smiles_vec_df = pd.DataFrame(smiles_vec, columns=smiles_vec_names, index=data.index)
        data = pd.concat([data, smiles_vec_df], axis=1)
        for name in smiles_vec_names:
            if name not in numericalFeatures:
                numericalFeatures.append(name)

for role in ['COOH', 'Amine']:
    xyz_col = f"{role} optXYZ"
    if xyz_col in data.columns:
        parsed_col = data[xyz_col].apply(parse_xyz_string)
        desc_3d = parsed_col.apply(compute_3d_descriptors).apply(pd.Series)
        desc_3d.columns = [f"{role}_3D_{n}" for n in ['Rg', 'MeanDist', 'StdDist', 'Inertia1', 'Inertia2']]
        data = pd.concat([data, desc_3d], axis=1)
        for col in desc_3d.columns:
            if col not in numericalFeatures:
                numericalFeatures.append(col)

for n in ['Rg', 'MeanDist', 'StdDist', 'Inertia1', 'Inertia2']:
    col1 = f"COOH_3D_{n}"
    col2 = f"Amine_3D_{n}"
    if col1 in data.columns and col2 in data.columns:
        data[f'COOHxAmine_3D_{n}_prod'] = data[col1] * data[col2]
        data[f'COOH-Amine_3D_{n}_diff'] = data[col1] - data[col2]
        data[f'COOH+Amine_3D_{n}_sum'] = data[col1] + data[col2]
        for feat in [f'COOHxAmine_3D_{n}_prod', f'COOH-Amine_3D_{n}_diff', f'COOH+Amine_3D_{n}_sum']:
            if feat not in numericalFeatures:
                numericalFeatures.append(feat)

for smilesCol, prefix in [('COOH SMILES', 'COOH'), ('Amine SMILES', 'Amine')]:
    if smilesCol in data.columns:
        soap_mat = np.stack(data[smilesCol].apply(get_soap))
        soap_cols = [f"{prefix}_SOAP_{i}" for i in range(soap_mat.shape[1])]
        soap_df = pd.DataFrame(soap_mat, columns=soap_cols, index=data.index)
        data = pd.concat([data, soap_df], axis=1)
        for name in soap_cols:
            if name not in numericalFeatures:
                numericalFeatures.append(name)

data = data.reset_index(drop=True)
data['yield_bin'] = pd.qcut(data['Yield'], q=3, labels=['low', 'med', 'high'])
categoricalFeatures_with_bin = categoricalFeatures + ['yield_bin']
-
if 'Patent' in data.columns:
    data['Patent_freq'] = data['Patent'].map(data['Patent'].value_counts())
    if 'Patent_freq' not in numericalFeatures:
        numericalFeatures.append('Patent_freq')

availableNumericalFeatures = [col for col in numericalFeatures if col in data.columns]
availableCategoricalFeatures = [col for col in categoricalFeatures_with_bin if col in data.columns]
X = data[availableNumericalFeatures + availableCategoricalFeatures]
y = data['Yield']
X = pd.get_dummies(X, columns=availableCategoricalFeatures, dummy_na=False)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, min_samples_leaf=3, subsample=0.8, max_features='sqrt', random_state=42)
model.fit(xTrain, yTrain)

yPred = model.predict(X)
data['yPred'] = yPred
data['yActual'] = y

if 'Patent' in data.columns:
    patents = data['Patent'].dropna().unique()
    for patent in patents:
        patent_data = data[data['Patent'] == patent]
        if len(patent_data) < 2:
            continue
        r2 = r2_score(patent_data['yActual'], patent_data['yPred'])
        plt.figure(figsize=(6,6))
        plt.scatter(patent_data['yActual'], patent_data['yPred'], alpha=0.7)
        plt.plot([patent_data['yActual'].min(), patent_data['yActual'].max()],
                 [patent_data['yActual'].min(), patent_data['yActual'].max()], 'r--')
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title(f"Patent {patent}: Actual vs Predicted Yield\nR² = {r2:.2f}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
