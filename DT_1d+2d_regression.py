import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

files = [f"/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File{i}_Done.csv" for i in range(1, 12)]
data_list = [pd.read_csv(file) for file in files]
combined_data = pd.concat(data_list, ignore_index=True)

numerical_features = ["Time", "Temperature", "COOH MW", "COOH logP", "Amine MW", "Amine logP"]
smiles_columns = ["Solvent SMILES", "Coupling Agent SMILES", "COOH SMILES", "Amine SMILES", "Additive SMILES"]
target = "Yield"

combined_data[numerical_features] = combined_data[numerical_features].fillna(combined_data[numerical_features].median())
combined_data[target] = combined_data[target].clip(0, 100)

RDLogger.DisableLog('rdApp.*')

def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=int)
    except:
        return np.zeros(n_bits, dtype=int)

fp_features = []
for col in smiles_columns:
    fp_array = np.array([smiles_to_fingerprint(sm) for sm in combined_data[col].fillna('')])
    fp_df = pd.DataFrame(fp_array, columns=[f"{col}_fp_{i}" for i in range(fp_array.shape[1])])
    combined_data = pd.concat([combined_data, fp_df], axis=1)
    fp_features.extend(fp_df.columns)

all_features = numerical_features + fp_features
X = combined_data[all_features]
y = combined_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('fp', 'passthrough', fp_features)
    ]
)

model = DecisionTreeRegressor(random_state=42)

param_dist = {
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, cv=5, scoring='r2', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"Best Parameters from RandomizedSearchCV: {search.best_params_}")

y_pred = search.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
