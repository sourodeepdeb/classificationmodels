import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import ConvexHull
from google.colab import drive
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
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

def safeCalcDesc(smiles):
    mol = molFromSmiles(smiles)
    if mol is None:
        return [np.nan]*6
    try:
        return calcDesc(mol)
    except:
        return [np.nan]*6

def morganFp(smiles, nBits=128):
    mol = molFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    arr = np.zeros(nBits, dtype=int)
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
numericalFeatures = ['Temperature', 'Time', 'TempTimeInteraction', 'COOH MW', 'COOH logP', 'Amine MW', 'Amine logP']
categoricalFeatures = ['Solvent', 'Coupling Agent', 'COOH', 'Amine', 'Additive', 'Amine SMILES', 'COOH SMILES']

for prefix in ['Amine', 'COOH']:
    xyzCol = f'{prefix} optXYZ'
    if xyzCol in data.columns:
        descDf = data[xyzCol].apply(parseXyzToDesc).apply(pd.Series)
        descDf.columns = [f'{prefix}_{col}' for col in descDf.columns]
        data = pd.concat([data, descDf], axis=1)
        for col in descDf.columns:
            if col not in numericalFeatures:
                numericalFeatures.append(col)

for smilesCol, prefix in smilesRoles:
    if smilesCol in data.columns:
        descDf = data[smilesCol].apply(safeCalcDesc).apply(pd.Series)
        descDf.columns = [f'{prefix}_{n}' for n in descNames]
        data = pd.concat([data, descDf], axis=1)
        for n in descNames:
            col = f'{prefix}_{n}'
            if col not in numericalFeatures:
                numericalFeatures.append(col)
        fps = data[smilesCol].fillna('').apply(lambda s: morganFp(s, nBits=fpBits))
        fpDf = pd.DataFrame(fps.tolist(), columns=[f'{prefix}_FP_{i}' for i in range(fpBits)], index=data.index)
        data = pd.concat([data, fpDf], axis=1)
        for i in range(fpBits):
            col = f'{prefix}_FP_{i}'
            if col not in numericalFeatures:
                numericalFeatures.append(col)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,4), min_df=1, max_features=32)
        smilesVec = vectorizer.fit_transform(data[smilesCol].fillna('')).toarray()
        smilesVecNames = [f"{prefix}_SMILES_{f}" for f in vectorizer.get_feature_names_out()]
        smilesVecDf = pd.DataFrame(smilesVec, columns=smilesVecNames, index=data.index)
        data = pd.concat([data, smilesVecDf], axis=1)
        for name in smilesVecNames:
            if name not in numericalFeatures:
                numericalFeatures.append(name)

availableNumericalFeatures = [col for col in numericalFeatures if col in data.columns]
availableCategoricalFeatures = [col for col in categoricalFeatures if col in data.columns]

X = data[availableNumericalFeatures+availableCategoricalFeatures]
y = data['Yield']

X = pd.get_dummies(X, columns=availableCategoricalFeatures, dummy_na=False)

X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

paramDist = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'C': np.logspace(-3, 3, 7),
    'gamma': ['scale', 'auto'] + list(np.logspace(-4,1,6))
}

search = RandomizedSearchCV(
    SVR(),
    param_distributions=paramDist,
    n_iter=25,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(xTrain, yTrain)

bestSVR = search.best_estimator_
yPred = bestSVR.predict(xTest)
r2 = r2_score(yTest, yPred)

print(f"SVR r^2: {r2}")
print(f"Best SVR params: {search.best_params_}")
