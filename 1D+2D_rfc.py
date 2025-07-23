import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
    if filename.endswith(".csv") and "MedChem_AMIDE_with_optXYZ" not in filename:
        df = pd.read_csv(os.path.join(pathToDir, filename))
        if "Reaction ID" in df.columns:
            df = df.drop(columns=["Reaction ID"])
        allData = pd.concat([allData, df], ignore_index=True)

data = allData.copy()
data = data[data['Yield'].notna() & data['Coupling Agent'].notna() & (data['Yield'] < 100) & (data['Yield'] != 0)]
data = data[data['Coupling Agent'].isin(CATEGORY_MAP.keys())]
data['Category'] = data['Coupling Agent'].map(CATEGORY_MAP)
data['TempTimeInteraction'] = data['Temperature'] * data['Time']

def molFromSmiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
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
data = data.reset_index(drop=True)
numerical_features = ['Temperature', 'Time', 'TempTimeInteraction']

for smiles_col, prefix in smiles_roles:
    if smiles_col not in data.columns:
        continue
    mols = data[smiles_col].apply(molFromSmiles)
    desc_top = mols.apply(lambda m: [
        Descriptors.MolWt(m) if m else np.nan,
        Descriptors.MolLogP(m) if m else np.nan
    ]).apply(pd.Series)
    desc_top.columns = [f'{prefix} MW', f'{prefix} logP']
    data = pd.concat([data, desc_top], axis=1)
    numerical_features += [f'{prefix} MW', f'{prefix} logP']
    fps = data[smiles_col].fillna('').apply(morgan_fp)
    fp_df = pd.DataFrame(fps.tolist(), columns=[f'{prefix}_FP_{i}' for i in range(2048)])
    data = pd.concat([data, fp_df], axis=1)
    numerical_features += list(fp_df.columns)

categorical_features = ['Solvent']
X = data[numerical_features + categorical_features]
y = data['Category']
X = pd.get_dummies(X, columns=categorical_features)
X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(xTrain, yTrain)
yPred = clf.predict(xTest)

print("Accuracy:", accuracy_score(yTest, yPred))
print("Classification Report:")
print(classification_report(yTest, yPred))

conf_matrix = confusion_matrix(yTest, yPred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Coupling Agent Category Prediction Confusion Matrix")
plt.show()

yScore = clf.predict_proba(xTest)
plt.figure(figsize=(8, 8))
classes = np.unique(yTrain)
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(yTest == classes[i], yScore[:, i])
    roc_auc = roc_auc_score(yTest == classes[i], yScore[:, i])
    plt.plot(fpr, tpr, label=f"Category {classes[i]} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC=0.5)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right")
plt.show()

train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()
