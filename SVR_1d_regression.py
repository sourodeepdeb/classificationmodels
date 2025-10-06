import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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
for col in smiles_columns:
    combined_data[col] = combined_data[col].fillna('').astype(str)
combined_data[target] = combined_data[target].clip(0, 100)

combined_data['Combined_SMILES'] = combined_data[smiles_columns].agg(' '.join, axis=1)

X = combined_data[numerical_features + ['Combined_SMILES']]
y = combined_data[target]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('smiles', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=1000), 'Combined_SMILES')
    ]
)

model = SVR()

param_dist = {
    'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'model__C': np.logspace(-3, 3, 7),
    'model__gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 6))
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=25,
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

print(f"\nSVR Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")
