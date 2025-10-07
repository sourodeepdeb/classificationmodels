import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer

from google.colab import drive
drive.mount('/content/drive')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

if 'Temperature' in data.columns and 'Time' in data.columns:
    data['TempTimeInteraction'] = data['Temperature'] * data['Time']

def molFromSmiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def calcDesc(mol):
    if mol is None:
        return [np.nan]*6
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

def morgan_fp(smiles, n_bits=128):
    mol = molFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

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

for smilesCol, prefix in smilesRoles:
    if smilesCol in data.columns:
        molCol = f'{prefix}_Mol'
        data[molCol] = data[smilesCol].apply(molFromSmiles)
        descDf = data[molCol].apply(calcDesc).apply(pd.Series)
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


availableNumericalFeatures = [col for col in numericalFeatures if col in data.columns]
availableCategoricalFeatures = [col for col in categoricalFeatures if col in data.columns]


X = data[availableNumericalFeatures + availableCategoricalFeatures]
y = data['Yield']


X = pd.get_dummies(X, columns=availableCategoricalFeatures, dummy_na=False)


imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)


xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)


param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}


search = RandomizedSearchCV(
    RandomForestRegressor(random_state=0),
    param_dist,
    n_iter=30,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
search.fit(xTrain, yTrain)


best_rf = search.best_estimator_
yPred = best_rf.predict(xTest)
r2 = r2_score(yTest, yPred)
print(f"r^2: {r2}")
print(f"Best hyperparameters: {search.best_params_}")
