import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception as e:
        print(f"Error parsing SMILES: {smiles} | {e}")
        return np.zeros(n_bits)

fp_features = []
for col in smiles_columns:
    print(f"Processing SMILES for {col}...")
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

model = BaggingRegressor(random_state=42)

param_dist = {
    'model__n_estimators': [100, 150, 200],
    'model__max_samples': [0.7, 1.0, 1.3],
    'model__max_features': [0.2, 0.5],
    'model__bootstrap': [False]
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='r2',
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print(f"Best Parameters from RandomizedSearchCV: {random_search.best_params_}")

y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nBagging Regressor Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

cv_score = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"Cross-validated R^2 Score: {np.mean(cv_score):.4f}")
