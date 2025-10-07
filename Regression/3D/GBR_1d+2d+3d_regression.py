import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import ConvexHull
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

def morganFp(smiles, nBits=128):
    mol = molFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def parseXyzToDesc(xyzStr):
    if not isinstance(xyzStr, str):
        return {}
    lines = xyzStr.strip().split('\n')
    coords = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                x, y, z = map(float, parts[1:4])
                coords.append((x, y, z))
            except:
                continue
    if len(coords) < 3:
        return {}
    coords = np.array(coords)
    centroid = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    radiusGyration = np.sqrt(np.mean(dists**2))
    inertia = np.cov(coords.T)
    eigvals = np.linalg.eigvalsh(inertia)
    pmi1, pmi2, pmi3 = eigvals
    npr1 = (pmi1 - pmi2) / pmi3 if pmi3 != 0 else np.nan
    npr2 = (2 * pmi2 - pmi1 - pmi3) / (pmi1 - pmi3) if (pmi1 - pmi3) != 0 else np.nan
    asphericity = (pmi1 - pmi2)**2 + (pmi1 - pmi3)**2 + (pmi2 - pmi3)**2
    asphericity /= 2 * (pmi1 + pmi2 + pmi3)**2 if (pmi1 + pmi2 + pmi3) != 0 else np.nan
    eccentricity = np.sqrt(pmi1**2 + pmi2**2 + pmi3**2 - pmi1*pmi2 - pmi1*pmi3 - pmi2*pmi3)
    eccentricity /= (pmi1 + pmi2 + pmi3) if (pmi1 + pmi2 + pmi3) != 0 else np.nan
    projXy = coords[:,:2]
    projXz = coords[:,[0,2]]
    projYz = coords[:,1:]
    try:
        areaXy = ConvexHull(projXy).volume
        areaXz = ConvexHull(projXz).volume
        areaYz = ConvexHull(projYz).volume
        totalArea = areaXy + areaXz + areaYz
        shadowXy = areaXy / totalArea if totalArea != 0 else np.nan
        shadowXz = areaXz / totalArea if totalArea != 0 else np.nan
        shadowYz = areaYz / totalArea if totalArea != 0 else np.nan
    except:
        shadowXy = shadowXz = shadowYz = np.nan
    return {
        'radiusGyration': radiusGyration,
        'PMI1': pmi1,
        'PMI2': pmi2,
        'PMI3': pmi3,
        'NPR1': npr1,
        'NPR2': npr2,
        'asphericity': asphericity,
        'eccentricity': eccentricity,
        'shadowXy': shadowXy,
        'shadowXz': shadowXz,
        'shadowYz': shadowYz
    }

descNames = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
smilesRoles = [
    ('COOH SMILES', 'COOH'),
    ('Amine SMILES', 'Amine'),
    ('Additive SMILES', 'Additive'),
    ('Coupling Agent SMILES', 'Coupling Agent'),
    ('Solvent SMILES', 'Solvent')
]
fpBits = 128
numericalFeatures = [
    'Temperature', 'Time', 'TempTimeInteraction',
    'COOH MW', 'COOH logP', 'Amine MW', 'Amine logP'
]
categoricalFeatures = [
    'Solvent', 'Coupling Agent', 'COOH', 'Amine', 'Additive', "Amine SMILES", "COOH SMILES"
]

for prefix in ['Amine', 'COOH']:
    xyzCol = f'{prefix} optXYZ'
    if xyzCol in data.columns:
        descDf = data[xyzCol].apply(parseXyzToDesc).apply(pd.Series)
        descDf.columns = [f'{prefix}_{col}' for col in descDf.columns]
        data = pd.concat([data, descDf], axis=1)
        for col in descDf.columns:
            if f'{prefix}_{col}' not in numericalFeatures:
                numericalFeatures.append(f'{prefix}_{col}')

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
        fps = data[smilesCol].fillna('').apply(lambda s: morganFp(s, nBits=fpBits))
        fpDf = pd.DataFrame(fps.tolist(), columns=[f'{prefix}_FP_{i}' for i in range(fpBits)], index=data.index)
        data = pd.concat([data, fpDf], axis=1)
        for i in range(fpBits):
            col = f'{prefix}_FP_{i}'
            if col in data.columns and col not in numericalFeatures:
                numericalFeatures.append(col)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), min_df=1, max_features=32)
        smilesVec = vectorizer.fit_transform(data[smilesCol].fillna('')).toarray()
        smilesVecNames = [f"{prefix}_SMILES_{f}" for f in vectorizer.get_feature_names_out()]
        smilesVecDf = pd.DataFrame(smilesVec, columns=smilesVecNames, index=data.index)
        data = pd.concat([data, smilesVecDf], axis=1)
        for name in smilesVecNames:
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

gbr_fs = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=3,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
gbr_fs.fit(xTrain, yTrain)
importances = gbr_fs.feature_importances_

N = 100
indices = np.argsort(importances)[::-1][:N]
xTrain_sel = xTrain[:, indices]
xTest_sel = xTest[:, indices]

param_dist = {
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [5, 7, 9],
    'min_samples_leaf': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'max_features': ['sqrt', 'log2', None],
    'n_estimators': [1000, 2000, 3000]
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=0),
    param_dist,
    n_iter=30,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
search.fit(xTrain_sel, yTrain)

best_gbr = search.best_estimator_
yPred = best_gbr.predict(xTest_sel)
r2 = r2_score(yTest, yPred)
print(f"r^2: {r2}")
print(f"Best hyperparameters: {search.best_params_}")
