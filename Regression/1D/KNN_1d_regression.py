import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import gc
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class YieldPredictor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.selector = SelectKBest(mutual_info_regression, k=50)
        self.model = None
        self.feature_names = None

    def load_data(self):
        print("Loading data...")
        file_paths = [
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File1_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File2_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File3_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File4_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File5_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File6_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File7_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File8_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File9_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File10_Done.csv',
            '/content/drive/Shared drives/MLDD/ORD_Data/CSV_FINAL_DATA_3884Datapoints_04_13_25/V2_File11_Done.csv',
        ]

        dfs = [pd.read_csv(path) for path in file_paths]
        full_df = pd.concat(dfs, ignore_index=True)

        full_df = full_df[(full_df['Yield'] > 0) & (full_df['Yield'] <= 100)].copy()

        if 'Temperature' in full_df.columns and 'Time' in full_df.columns:
            full_df['TempTimeInteraction'] = full_df['Temperature'] * full_df['Time']

        train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=0)
        return train_df, test_df

    def extract_features(self, df, is_train=True):
        print("Extracting features...")

        num_features = ['Temperature', 'Time', 'TempTimeInteraction']

        desc_features = []
        for col, prefix in [('COOH SMILES', 'COOH'), ('Amine SMILES', 'Amine')]:
            if col in df.columns:
                df[f'{prefix}_Mol'] = df[col].apply(Chem.MolFromSmiles)
                df[f'{prefix}_MW'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.MolWt(m) if m else np.nan)
                df[f'{prefix}_LogP'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.MolLogP(m) if m else np.nan)
                df[f'{prefix}_HBA'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.NumHAcceptors(m) if m else np.nan)
                desc_features.extend([f'{prefix}_MW', f'{prefix}_LogP', f'{prefix}_HBA'])
                df.drop(f'{prefix}_Mol', axis=1, inplace=True)

        cat_features = ['Solvent', 'Coupling Agent']
        features = num_features + desc_features + cat_features
        features = [f for f in features if f in df.columns]

        if is_train:
            self.feature_names = features

        return df, features

    def preprocess_data(self, train_df, test_df):
        print("Preprocessing data...")

        train_df, features = self.extract_features(train_df, is_train=True)
        test_df, _ = self.extract_features(test_df, is_train=False)

        X_train = train_df[features]
        y_train = train_df['Yield']
        X_test = test_df[features]
        y_test = test_df['Yield']

        categorical_cols = [f for f in features if f in ['Solvent', 'Coupling Agent']]
        if categorical_cols:
            X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

            missing_cols = set(X_train.columns) - set(X_test.columns)
            for col in missing_cols:
                X_test[col] = 0
            X_test = X_test[X_train.columns]

        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if hasattr(self, 'selector'):
            X_train = self.selector.fit_transform(X_train, y_train)
            X_test = self.selector.transform(X_test)

        gc.collect()
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        print("Training model...")

        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['distance'],
            'p': [1],
            'algorithm': ['auto'],
            'leaf_size': [30]
        }

        knn = KNeighborsRegressor(n_jobs=-1)

        search = RandomizedSearchCV(
            knn,
            param_grid,
            n_iter=5,
            cv=2,
            scoring='r2',
            verbose=1
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_

        print("\n=== Best Model Parameters ===")
        print(search.best_params_)

        return self.model

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Test R2 Score: {r2:}")
        return r2

    def run_pipeline(self):
        train_df, test_df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(train_df, test_df)
        self.train_model(X_train, y_train)
        self.evaluate(X_test, y_test)

if __name__ == "__main__":
    predictor = YieldPredictor()
    predictor.run_pipeline()
