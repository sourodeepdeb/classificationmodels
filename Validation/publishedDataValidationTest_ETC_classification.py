import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

COUPLING_AGENT_CATEGORIES = {
    1: ["DCC", "DIC", "EDC", "CDI", "DIPC", "N,N′-Ethylcarbodiimide hydrochloride"],
    2: ["HBTU", "HATU", "HCTU", "TBTU", "TATU", "TOTU", "COMU", "TDBTU", "TSTU", "TNTU", "TPTU", "DEPBT"],
    3: ["BOP", "PyBOP", "PyAOP", "PyBrOP", "BOP-Cl"],
    4: ["HOBt", "Benzotriazole", "BTZ", "Trifluoromethanesulfonic acid"],
    5: ["TCFH", "EEDQ", "1-Cyano-1-methyl-2-oxo-2H-pyrido[1,2-a]pyrimidine"]
}
CATEGORY_MAP = {agent: cat for cat, agents in COUPLING_AGENT_CATEGORIES.items() for agent in agents}
CATEGORY_AGENT_REP = {cat: agents[0] for cat, agents in COUPLING_AGENT_CATEGORIES.items()}

def molFromSmiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Chem.SanitizeMol(mol)
            AllChem.Compute2DCoords(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
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

smiles_roles = [('COOH SMILES', 'COOH'), ('Amine SMILES', 'Amine')]

path_to_no_patent_dir = "/content/drive/MyDrive/ordMLFiles"
allData = pd.DataFrame()
for filename in os.listdir(path_to_no_patent_dir):
    if filename.endswith("_no_patent.csv"):
        df = pd.read_csv(os.path.join(path_to_no_patent_dir, filename))
        if "Reaction ID" in df.columns:
            df = df.drop(columns=["Reaction ID"])
        allData = pd.concat([allData, df], ignore_index=True)

data = allData.copy()
data = data.dropna(subset=['Yield', 'Coupling Agent', 'Temperature', 'Time'])
data = data[(data['Yield'] < 100) & (data['Yield'] != 0)]
data = data[data['Coupling Agent'].isin(CATEGORY_MAP.keys())]
data['Category'] = data['Coupling Agent'].map(CATEGORY_MAP)
data['TempTimeInteraction'] = data['Temperature'] * data['Time']

numerical_features = ['Temperature', 'Time', 'TempTimeInteraction']
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
    desc_1d.columns = [f'{prefix}_MW', f'{prefix}_logP', f'{prefix}_HAcceptors', f'{prefix}_HDonors', f'{prefix}_TPSA', f'{prefix}_RotBonds', f'{prefix}_RingCount', f'{prefix}_FracCSP3', f'{prefix}_AroRings']
    data = pd.concat([data, desc_1d], axis=1)
    numerical_features += list(desc_1d.columns)
    fps = data[smiles_col].fillna('').apply(morgan_fp)
    fp_df = pd.DataFrame(fps.tolist(), columns=[f'{prefix}_FP_{i}' for i in range(2048)])
    data = pd.concat([data, fp_df], axis=1)
    numerical_features += list(fp_df.columns)

categorical_features = ['Solvent']
X = data[numerical_features + categorical_features]
y = data['Category']
X = pd.get_dummies(X, columns=categorical_features)
X = X.dropna()
y = y.loc[X.index]

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)
xTrain, xTest, yTrain, yTest = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

clf = ExtraTreesClassifier(
    n_estimators=1500,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced_subsample')
clf.fit(xTrain, yTrain)

yPred = clf.predict(xTest)
yScore = clf.predict_proba(xTest)
print("Accuracy on No-Patent Data:", accuracy_score(yTest, yPred))
print("Classification Report on No-Patent Data:")
print(classification_report(yTest, yPred))
sns.heatmap(confusion_matrix(yTest, yPred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("No-Patent Confusion Matrix")
plt.show()
plt.figure(figsize=(8, 8))
classes = np.unique(yTrain)
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(yTest == classes[i], yScore[:, i])
    roc_auc = roc_auc_score(yTest == classes[i], yScore[:, i])
    plt.plot(fpr, tpr, label=f"Category {classes[i]} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("No-Patent ROC Curves")
plt.legend(loc="lower right")
plt.show()

cleaned_csv = "/content/drive/MyDrive/ordMLFiles/MLDD_Sheet1_cleaned.csv"
new_data = pd.read_csv(cleaned_csv)
dropcols = [c for c in ['Reaction ID', 'patent', 'Patent'] if c in new_data.columns]
new_data = new_data.drop(columns=dropcols, errors='ignore')
new_data = new_data.dropna(subset=['Yield', 'Coupling Agent', 'Temperature', 'Time'])
new_data['Category'] = new_data['Coupling Agent'].map(CATEGORY_MAP)
new_data = new_data[new_data['Category'].notna()]
new_data['TempTimeInteraction'] = new_data['Temperature'] * new_data['Time']

numerical_features_new = ['Temperature', 'Time', 'TempTimeInteraction']
for smiles_col, prefix in smiles_roles:
    if smiles_col not in new_data.columns:
        continue
    mols = new_data[smiles_col].apply(molFromSmiles)
    desc_1d_new = mols.apply(lambda m: [
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
    desc_1d_new.columns = [f'{prefix}_MW', f'{prefix}_logP', f'{prefix}_HAcceptors', f'{prefix}_HDonors', f'{prefix}_TPSA', f'{prefix}_RotBonds', f'{prefix}_RingCount', f'{prefix}_FracCSP3', f'{prefix}_AroRings']
    new_data = pd.concat([new_data, desc_1d_new], axis=1)
    numerical_features_new += list(desc_1d_new.columns)
    fps_new = new_data[smiles_col].fillna('').apply(morgan_fp)
    fp_df_new = pd.DataFrame(fps_new.tolist(), columns=[f'{prefix}_FP_{i}' for i in range(2048)], index=new_data.index)
    new_data = pd.concat([new_data, fp_df_new], axis=1)
    numerical_features_new += list(fp_df_new.columns)

X_new = new_data[numerical_features_new + categorical_features]
X_new = pd.get_dummies(X_new, columns=categorical_features)
X_new = X_new.reindex(columns=X.columns, fill_value=0)
X_new = X_new.dropna()
X_new_imp = imputer.transform(X_new)
X_new_scaled = scaler.transform(X_new_imp)
y_new_pred = clf.predict(X_new_scaled)
yScore_new = clf.predict_proba(X_new_scaled)
print("Accuracy on Cleaned CSV:", accuracy_score(new_data['Category'].loc[X_new.index], y_new_pred))
print("Classification Report on Cleaned CSV:")
print(classification_report(new_data['Category'].loc[X_new.index], y_new_pred))
sns.heatmap(confusion_matrix(new_data['Category'].loc[X_new.index], y_new_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Cleaned Confusion Matrix")
plt.show()
plt.figure(figsize=(8, 8))
classes_new = np.unique(new_data['Category'].loc[X_new.index])
for i in range(len(classes_new)):
    fpr, tpr, _ = roc_curve(new_data['Category'].loc[X_new.index] == classes_new[i], yScore_new[:, i])
    roc_auc = roc_auc_score(new_data['Category'].loc[X_new.index] == classes_new[i], yScore_new[:, i])
    plt.plot(fpr, tpr, label=f"Category {classes_new[i]} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Cleaned ROC Curves")
plt.legend(loc="lower right")
plt.show()
