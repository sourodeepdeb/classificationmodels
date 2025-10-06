import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, rdMolDescriptors, rdPartialCharges, rdDistGeom
from rdkit.Chem.Pharm3D import Pharmacophore
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import pdist, squareform

COUPLING_AGENT_CATEGORIES = {
    1: ["DCC", "DIC", "EDC", "CDI", "DIPC", "N,N′-Ethylcarbodiimide hydrochloride"],
    2: ["HBTU", "HATU", "HCTU", "TBTU", "TATU", "TOTU", "COMU", "TDBTU", "TSTU", "TNTU", "TPTU", "DEPBT"],
    3: ["BOP", "PyBOP", "PyAOP", "PyBrOP", "BOP-Cl"],
    4: ["HOBt", "Benzotriazole", "BTZ", "Trifluoromethanesulfonic acid"],
    5: ["TCFH", "EEDQ", "1-Cyano-1-methyl-2-oxo-2H-pyrido[1,2-a]pyrimidine"]
}
CATEGORY_MAP = {agent: cat for cat, agents in COUPLING_AGENT_CATEGORIES.items() for agent in agents}

pathToDir = "/content/drive/MyDrive/ordMLFiles"
allData = pd.DataFrame()
for filename in os.listdir(pathToDir):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(pathToDir, filename))
        if "Reaction ID" in df.columns:
            df = df.drop(columns=["Reaction ID"])
        allData = pd.concat([allData, df], ignore_index=True)

data = allData.copy()
data = data[data['Yield'].notna() & data['Coupling Agent'].notna() & (data['Yield'] < 100) & (data['Yield'] != 0)]
data = data[data['Coupling Agent'].isin(CATEGORY_MAP.keys())]
data['Category'] = data['Coupling Agent'].map(CATEGORY_MAP)
data['TempTimeInteraction'] = data['Temperature'] * data['Time']

if 'Patent' in data.columns:
    data['Patent_freq'] = data['Patent'].map(data['Patent'].value_counts())

def molFromSmiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Chem.SanitizeMol(mol)
            AllChem.Compute2DCoords(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        return mol
    except:
        return None

def morgan_fp(smiles, n_bits=2048, radius=2):
    mol = molFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_3d_features(mol):
    if mol is None:
        return [np.nan] * 50
    try:
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=ps)
        for cid in cids:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        features = []
        props = rdMolDescriptors.Properties()
        features.extend(props.ComputeProperties(mol))
        factory = Pharmacophore.PharmacophoreFactory()
        featParams = Pharmacophore.PharmacophoreFeatureParams()
        feats = factory.GetFeaturesForMol(mol, featParams)
        features.extend([
            len(feats),
            sum(1 for f in feats if f.GetFamily() == 'Donor'),
            sum(1 for f in feats if f.GetFamily() == 'Acceptor'),
            sum(1 for f in feats if f.GetFamily() == 'Aromatic'),
            sum(1 for f in feats if f.GetFamily() == 'Hydrophobe')
        ])
        vol_features = []
        sa_features = []
        pmis = []
        nprs = []
        rgyr = []
        dipole_moments = []
        for cid in cids:
            vol_features.append(AllChem.ComputeMolVolume(mol, confId=cid))
            sa_features.append(rdMolDescriptors.CalcLabuteASA(mol, confId=cid))
            pmis.append(rdMolDescriptors.CalcPMI1(mol, confId=cid))
            nprs.append(rdMolDescriptors.CalcNPR1(mol, confId=cid))
            rgyr.append(rdMolDescriptors.CalcRadiusOfGyration(mol, confId=cid))
            dipole_moments.append(AllChem.ComputeDipoleMoment(mol, confId=cid))
        features.extend([
            np.mean(vol_features), np.std(vol_features),
            np.mean(sa_features), np.std(sa_features),
            np.mean(pmis), np.std(pmis),
            np.mean(nprs), np.std(nprs),
            np.mean(rgyr), np.std(rgyr),
            np.mean(dipole_moments), np.std(dipole_moments)
        ])
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        features.extend([
            np.mean(charges), np.std(charges),
            np.max(charges), np.min(charges),
            rdMolDescriptors.CalcNumAtomStereoCenters(mol),
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        ])
        shape_descriptors = [
            rdMolDescriptors.CalcAsphericity(mol),
            rdMolDescriptors.CalcEccentricity(mol),
            rdMolDescriptors.CalcInertialShapeFactor(mol),
            rdMolDescriptors.CalcSpherocityIndex(mol),
            rdMolDescriptors.CalcPBF(mol)
        ]
        features.extend(shape_descriptors)
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        if ff:
            features.append(ff.CalcEnergy())
        else:
            features.append(np.nan)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        if len(pos) > 1:
            dist_matrix = squareform(pdist(pos))
            features.extend([
                np.mean(dist_matrix), np.std(dist_matrix),
                np.max(dist_matrix), np.min(dist_matrix[np.nonzero(dist_matrix)])
            ])
        else:
            features.extend([np.nan]*4)
        return features
    except:
        return [np.nan] * 50

smiles_roles = [('COOH SMILES', 'COOH'), ('Amine SMILES', 'Amine')]
data = data.reset_index(drop=True)
numerical_features = ['Temperature', 'Time', 'TempTimeInteraction', 'Patent_freq']  

for smiles_col, prefix in smiles_roles:
    if smiles_col not in data.columns:
        continue
    mols = data[smiles_col].apply(molFromSmiles)
    desc_1d = mols.apply(lambda m: [
        Descriptors.MolWt(m) if m else np.nan,
        Descriptors.MolLogP(m) if m else np.nan,
        Descriptors.NumHAcceptors(m) if m else np.nan,
        Descriptors.NumHDonors(m) if m else np.nan,
        Descriptors.TPSA(m) if m else np.nan,
        Descriptors.NumRotatableBonds(m) if m else np.nan,
        Descriptors.RingCount(m) if m else np.nan,
        Descriptors.FractionCSP3(m) if m else np.nan,
        Descriptors.NumAromaticRings(m) if m else np.nan
    ]).apply(pd.Series)
    desc_1d.columns = [f'{prefix}_MW', f'{prefix}_logP', f'{prefix}_HAcceptors', f'{prefix}_HDonors',
                      f'{prefix}_TPSA', f'{prefix}_RotBonds', f'{prefix}_RingCount', f'{prefix}_FracCSP3', f'{prefix}_AroRings']
    data = pd.concat([data, desc_1d], axis=1)
    numerical_features += list(desc_1d.columns)
    fps = data[smiles_col].fillna('').apply(morgan_fp)
    fp_df = pd.DataFrame(fps.tolist(), columns=[f'{prefix}_FP_{i}' for i in range(2048)])
    data = pd.concat([data, fp_df], axis=1)
    numerical_features += list(fp_df.columns)
    desc_3d = data[smiles_col].apply(lambda x: get_3d_features(molFromSmiles(x))).apply(pd.Series)
    desc_3d.columns = [f'{prefix}_3D_{i}' for i in range(50)]
    data = pd.concat([data, desc_3d], axis=1)
    numerical_features += list(desc_3d.columns)

categorical_features = ['Solvent']
X = data[numerical_features + categorical_features]
y = data['Category']
X = pd.get_dummies(X, columns=categorical_features)
X = X.apply(pd.to_numeric, errors='coerce')
X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = ExtraTreesClassifier(
    n_estimators=1500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced_subsample'
)

clf.fit(xTrain, yTrain)
yPred = clf.predict(xTest)

print("Accuracy:", accuracy_score(yTest, yPred))
print("Classification Report:")
print(classification_report(yTest, yPred))

if 'Patent' in data.columns:
    patents = data['Patent'].dropna().unique()
    classes = np.unique(y)
    for patent in patents:
        patent_mask = data['Patent'] == patent
        X_patent = X[patent_mask]
        y_patent = y[patent_mask]

        if len(y_patent.unique()) < 2 or len(y_patent) < 3:
            continue

        y_pred_patent = clf.predict(X_patent)
        conf_matrix = confusion_matrix(y_patent, y_pred_patent, labels=classes)
        acc = accuracy_score(y_patent, y_pred_patent)

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Category")
        plt.ylabel("Actual Category")
        plt.title(f"Patent {patent} - Classification Heatmap\nSamples = {len(y_patent)}")

        plt.figtext(0.5, -0.05, f"Accuracy: {acc:.3f}", ha="center", fontsize=12)

        plt.tight_layout()
        plt.show()

yScore = clf.predict_proba(xTest)
plt.figure(figsize=(8, 8))
classes = np.unique(yTrain)
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(yTest == classes[i], yScore[:, i])
    roc_auc = roc_auc_score(yTest == classes[i], yScore[:, i])
    plt.plot(fpr, tpr, label=f"Category {classes[i]} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC=0.5)")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right")
plt.show()

importances = clf.feature_importances_
indices = np.argsort(importances)[-20:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
plt.xlabel("Relative Importance")
plt.title("Top 20 Important Features")
plt.show()
