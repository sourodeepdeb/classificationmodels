
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

files = [f"/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File{i}_Done.csv"
         for i in range(1, 12)]
data_list = [pd.read_csv(file) for file in files]
combined_data = pd.concat(data_list, ignore_index=True)

numerical_features = ["Time", "Temperature", "COOH MW", "COOH logP", "Amine MW", "Amine logP"]
smiles_columns = ["Solvent SMILES", "Coupling Agent SMILES", "COOH SMILES", "Amine SMILES", "Additive SMILES"]
target = "Yield"

combined_data[numerical_features] = combined_data[numerical_features].fillna(combined_data[numerical_features].median())
combined_data[target] = combined_data[target].clip(0, 100)

combined_data[smiles_columns] = combined_data[smiles_columns].fillna("")

X = combined_data[numerical_features + smiles_columns]
y = combined_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('smiles', OneHotEncoder(handle_unknown='ignore'), smiles_columns)
    ]
)

model = DecisionTreeRegressor(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

param_dist = {
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

print("Running RandomizedSearchCV for Decision Tree (SMILES + numeric, no fingerprints)...")
search.fit(X_train, y_train)
print("Hyperparameter search complete.\n")

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:")
print(search.best_params_, "\n")

print("Model Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")


