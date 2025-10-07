import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

files = [f"/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File{i}_Done.csv" for i in range(1, 12)]
data_list = [pd.read_csv(file) for file in files]
combined_data = pd.concat(data_list, ignore_index=True)

numerical_features = ["Time", "Temperature", "COOH MW", "COOH logP", "Amine MW", "Amine logP"]
smiles_columns = ["Solvent SMILES", "Coupling Agent SMILES", "COOH SMILES", "Amine SMILES", "Additive SMILES"]
target = "Yield"

combined_data[numerical_features] = combined_data[numerical_features].fillna(combined_data[numerical_features].median())

combined_data[target] = combined_data[target].clip(0, 100)

combined_data[numerical_features] = combined_data[numerical_features].fillna(combined_data[numerical_features].median())

for col in smiles_columns:
    combined_data[col] = combined_data[col].fillna('').astype(str)

combined_data[target] = combined_data[target].clip(0, 100)

X = combined_data[numerical_features + smiles_columns]
y = combined_data[target]

smiles_transformers = [
    (col, TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4)), col) for col in smiles_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        *smiles_transformers
    ],
    remainder='drop'
)

model = MLPRegressor(random_state=42, max_iter=200)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'model__hidden_layer_sizes': [(100,)],
    'model__activation': ['tanh'],
    'model__solver': ['adam'],
    'model__alpha': [0.01],
    'model__learning_rate': ['adaptive']
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=3,
    cv=2,
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

print(f"\nMLP Regressor Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")
