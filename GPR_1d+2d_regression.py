import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer

from google.colab import drive
drive.mount('/content/drive')

pathToDir = "/content/drive/MyDrive/ordMLFiles"

allData = pd.DataFrame()
for filename in os.listdir(pathToDir):
    if filename.endswith(".csv"):
        pathToFile = os.path.join(pathToDir, filename)
        data = pd.read_csv(pathToFile)
        allData = pd.concat([allData, data], ignore_index=True)

data = allData.copy()
data = data[data['Yield'] < 100]
data = data[data['Yield'] != 0]
data = data.replace('None', np.nan)
data = data.dropna(subset=['Yield'])

allColumns = data.columns.tolist()

numericalFeatures = [
    'Temperature', 'Time',
    'Reactant 1 MW', 'Reactant 1 LogP',
    'Reactant 2 MW', 'Reactant 2 LogP'
]

categoricalFeatures = [
    'Solvent 1 Name', 'Reactant 1 Name',
    'Reactant 2 Name'
]

def morgan_fp(smiles, n_bits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

smiles_columns = [col for col in data.columns if 'SMILES' in col]

for smiles_col in smiles_columns:
    fps = data[smiles_col].fillna('').apply(lambda s: morgan_fp(s, n_bits=256))
    fp_df = pd.DataFrame(fps.tolist(), columns=[f'{smiles_col}_FP_{i}' for i in range(256)], index=data.index)
    data = pd.concat([data, fp_df], axis=1)
    numericalFeatures += fp_df.columns.tolist()
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=32)
    smiles_vec = vectorizer.fit_transform(data[smiles_col].fillna('')).toarray()
    smiles_vec_names = [f"{smiles_col}_SMILES_{f}" for f in vectorizer.get_feature_names_out()]
    smiles_vec_df = pd.DataFrame(smiles_vec, columns=smiles_vec_names, index=data.index)
    data = pd.concat([data, smiles_vec_df], axis=1)
    numericalFeatures += smiles_vec_names

identifierFeatures = [col for col in allColumns if col not in numericalFeatures + categoricalFeatures + ['Yield']]

availableNumericalFeatures = [col for col in numericalFeatures if col in data.columns]
availableCategoricalFeatures = [col for col in categoricalFeatures if col in data.columns]
availableIdentifierFeatures = [col for col in identifierFeatures if col in data.columns]

for col in availableNumericalFeatures:
    data[col] = pd.to_numeric(data[col], errors='coerce')

numericalData = data[availableNumericalFeatures]
imputer = SimpleImputer(strategy='median')
numericalDataImputed = imputer.fit_transform(numericalData)
numericalData = pd.DataFrame(numericalDataImputed, columns=availableNumericalFeatures, index=data.index)

quantileTransformer = QuantileTransformer(output_distribution='normal', random_state=42)
numericalDataTransformed = quantileTransformer.fit_transform(numericalData)
numericalData = pd.DataFrame(numericalDataTransformed, columns=availableNumericalFeatures, index=data.index)

data['tempTimeInteraction'] = data['Temperature'] * data['Time']
if 'Reactant 1 MW' in data.columns and 'Reactant 2 MW' in data.columns:
    data['mwRatioR1R2'] = data['Reactant 1 MW'] / (data['Reactant 2 MW'] + 1e-9)
if 'Reactant 1 LogP' in data.columns and 'Reactant 2 LogP' in data.columns:
    data['logpDiffR1R2'] = data['Reactant 1 LogP'] - data['Reactant 2 LogP']

for col in availableCategoricalFeatures:
    data[col] = data[col].fillna('Unknown')
    counts = data[col].value_counts()
    rareCats = counts[counts < 5].index
    data[col] = data[col].apply(lambda x: 'Rare' if x in rareCats else x)

X = pd.get_dummies(data[availableNumericalFeatures + availableCategoricalFeatures + availableIdentifierFeatures],
                   columns=availableCategoricalFeatures + availableIdentifierFeatures, dummy_na=False)

if 'tempTimeInteraction' in data.columns:
    X['tempTimeInteraction'] = data['tempTimeInteraction']
if 'mwRatioR1R2' in data.columns:
    X['mwRatioR1R2'] = data['mwRatioR1R2']
if 'logpDiffR1R2' in data.columns:
    X['logpDiffR1R2'] = data['logpDiffR1R2']

y = data['Yield'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

kernel = ConstantKernel(89.9**2) * RBF(length_scale=429) + WhiteKernel(noise_level=0.00043)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, random_state=42)

gpr.fit(xTrain, yTrain)
yPred, yStd = gpr.predict(xTest, return_std=True)
r2 = r2_score(yTest, yPred)

print(f"Kernel used: {gpr.kernel_}")
print(f"r^2: {r2}")
