import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.rdMolTransforms import ComputePrincipalAxesAndMoments
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import gc
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import moment

from google.colab import drive
drive.mount('/content/drive')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class EnhancedYieldPredictor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.selector = SelectKBest(mutual_info_regression, k=100)
        self.model = None
        self.feature_names = None

    def load_data(self):
        print("Loading data...")
        file_paths = [f'/content/drive/My Drive/ordMLFiles/V2_File{i}_Done_with_optXYZ.csv' for i in range(1, 12)]
        dfs = []
        for path in file_paths:
            df = pd.read_csv(path)
            df = self.parse_xyz_data(df)
            dfs.append(df)
        full_df = pd.concat(dfs, ignore_index=True)
        full_df = full_df[(full_df['Yield'] > 0) & (full_df['Yield'] <= 100)].copy()

        if 'Temperature' in full_df.columns and 'Time' in full_df.columns:
            full_df['TempTimeInteraction'] = full_df['Temperature'] * full_df['Time']
            full_df['TempSquared'] = full_df['Temperature'] ** 2
            full_df['TimeSquared'] = full_df['Time'] ** 2

        full_df['yield_bin'] = pd.qcut(full_df['Yield'], q=3, labels=['low', 'med', 'high'])

        train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=0)
        return train_df, test_df

    def parse_xyz_data(self, df):
        print("Parsing XYZ data...")
        for prefix in ['COOH', 'AMINE']:
            xyz_col = f"{prefix} optXYZ"
            if xyz_col in df.columns:
                df[f'{prefix}_xyz'] = df[xyz_col].apply(self._parse_xyz_string)
                df[f'{prefix}_gyration_radius'] = df[f'{prefix}_xyz'].apply(self.calculate_gyration_radius)
                df[f'{prefix}_asphericity'] = df[f'{prefix}_xyz'].apply(self.calculate_asphericity)
                df[f'{prefix}_eccentricity'] = df[f'{prefix}_xyz'].apply(self.calculate_eccentricity)
                df[f'{prefix}_inertia_moments'] = df[f'{prefix}_xyz'].apply(self.calculate_inertia_moments)
                df[f'{prefix}_pmi_ratio'] = df[f'{prefix}_xyz'].apply(self.calculate_pmi_ratio)
                df[f'{prefix}_surface_area'] = df[f'{prefix}_xyz'].apply(self.estimate_surface_area)
        return df

    def _parse_xyz_string(self, xyz_str):
        if pd.isna(xyz_str):
            return []
        lines = xyz_str.strip().split('\n')
        coords = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except:
                    continue
        return np.array(coords)

    def calculate_gyration_radius(self, coords):
        if len(coords) == 0:
            return np.nan
        centroid = np.mean(coords, axis=0)
        squared_distances = np.sum((coords - centroid)**2, axis=1)
        return np.sqrt(np.mean(squared_distances))

    def calculate_asphericity(self, coords):
        if len(coords) < 3:
            return np.nan
        gyration_tensor = np.cov(coords.T)
        eigenvalues = np.linalg.eigvalsh(gyration_tensor)
        return eigenvalues[2] - 0.5 * (eigenvalues[0] + eigenvalues[1])

    def calculate_eccentricity(self, coords):
        if len(coords) < 2:
            return np.nan
        gyration_tensor = np.cov(coords.T)
        eigenvalues = np.linalg.eigvalsh(gyration_tensor)
        return np.sqrt(eigenvalues[2] - eigenvalues[0]) / np.sqrt(eigenvalues[2])

    def calculate_inertia_moments(self, coords):
        if len(coords) < 3:
            return [np.nan, np.nan, np.nan]
        masses = np.ones(len(coords))
        centroid = np.average(coords, axis=0, weights=masses)
        centered = coords - centroid
        inertia = np.zeros((3,3))
        for i in range(len(centered)):
            x, y, z = centered[i]
            inertia[0,0] += masses[i] * (y**2 + z**2)
            inertia[1,1] += masses[i] * (x**2 + z**2)
            inertia[2,2] += masses[i] * (x**2 + y**2)
            inertia[0,1] -= masses[i] * x * y
            inertia[0,2] -= masses[i] * x * z
            inertia[1,2] -= masses[i] * y * z
        inertia[1,0] = inertia[0,1]
        inertia[2,0] = inertia[0,2]
        inertia[2,1] = inertia[1,2]
        return np.linalg.eigvalsh(inertia)

    def calculate_pmi_ratio(self, coords):
        moments = self.calculate_inertia_moments(coords)
        if any(np.isnan(moments)):
            return np.nan
        return moments[0] / moments[1] if moments[1] != 0 else np.nan

    def estimate_surface_area(self, coords):
        if len(coords) < 4:
            return np.nan
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            return hull.area
        except:
            return np.nan

    def compute_morgan_fingerprints(self, smiles_series, radius=3, n_bits=256):
        fps = []
        for smiles in tqdm(smiles_series.fillna(''), desc="Computing fingerprints"):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                fps.append(fp)
            else:
                fps.append(np.zeros(n_bits, dtype=int))
        return np.array(fps)

    def extract_features(self, df, is_train=True):
        print("Extracting features...")

        num_features = ['Temperature', 'Time', 'TempTimeInteraction', 'TempSquared', 'TimeSquared']
        xyz_features = []
        for prefix in ['COOH', 'AMINE']:
            xyz_features.extend([
                f'{prefix}_gyration_radius',
                f'{prefix}_asphericity',
                f'{prefix}_eccentricity',
                f'{prefix}_pmi_ratio',
                f'{prefix}_surface_area'
            ])
            for i in range(3):
                df[f'{prefix}_inertia_{i}'] = df[f'{prefix}_inertia_moments'].apply(
                    lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan
                )
                xyz_features.append(f'{prefix}_inertia_{i}')

        fp_features = []
        for col in ['COOH SMILES', 'Amine SMILES']:
            if col in df.columns:
                fps = self.compute_morgan_fingerprints(df[col])
                fp_cols = [f'{col}_FP_{i}' for i in range(fps.shape[1])]
                fp_df = pd.DataFrame(fps, columns=fp_cols, index=df.index)
                fp_features.extend(fp_cols)
                df = pd.concat([df, fp_df], axis=1)

        desc_features = []
        for col, prefix in [('COOH SMILES', 'COOH'), ('Amine SMILES', 'Amine')]:
            if col in df.columns:
                df[f'{prefix}_Mol'] = df[col].apply(Chem.MolFromSmiles)
                df[f'{prefix}_MW'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.MolWt(m) if m else np.nan)
                df[f'{prefix}_LogP'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.MolLogP(m) if m else np.nan)
                df[f'{prefix}_HBA'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.NumHAcceptors(m) if m else np.nan)
                df[f'{prefix}_HBD'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.NumHDonors(m) if m else np.nan)
                df[f'{prefix}_TPSA'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.TPSA(m) if m else np.nan)
                df[f'{prefix}_NumRotatableBonds'] = df[f'{prefix}_Mol'].apply(lambda m: Descriptors.NumRotatableBonds(m) if m else np.nan)
                df[f'{prefix}_NumRings'] = df[f'{prefix}_Mol'].apply(lambda m: rdMolDescriptors.CalcNumRings(m) if m else np.nan)
                desc_features.extend([
                    f'{prefix}_MW', f'{prefix}_LogP', f'{prefix}_HBA', f'{prefix}_HBD',
                    f'{prefix}_TPSA', f'{prefix}_NumRotatableBonds', f'{prefix}_NumRings'
                ])
                df.drop(f'{prefix}_Mol', axis=1, inplace=True)

        cat_features = ['Solvent', 'Coupling Agent', 'yield_bin']
        features = num_features + xyz_features + fp_features + desc_features + cat_features
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

        categorical_cols = [f for f in features if f in ['Solvent', 'Coupling Agent', 'yield_bin']]
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
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [10, 20, 30, 40],
            'metric': ['minkowski', 'manhattan']
        }

        knn = KNeighborsRegressor(n_jobs=-1)

        search = RandomizedSearchCV(
            knn,
            param_grid,
            n_iter=50,
            cv=5,
            scoring='r2',
            verbose=1,
            random_state=0,
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_

        print("\n=== Best Model Parameters ===")
        print(search.best_params_)
        print(f"Best CV R2: {search.best_score_:.3f}")

        return search.best_estimator_

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Test R2 Score: {r2:.3f}")
        print(f"Test MAE: {mae:.2f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color='teal', edgecolors='k')
        plt.plot([0, 100], [0, 100], '--', color='gray', linewidth=2)
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title("Predicted vs Actual Yield (Test Set)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        return r2

    def run_pipeline(self):
        train_df, test_df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(train_df, test_df)
        self.train_model(X_train, y_train)
        return self.evaluate(X_test, y_test)

if __name__ == "__main__":
    predictor = EnhancedYieldPredictor()
    r2_score = predictor.run_pipeline()
    print(f"\nFinal Model R2 Score: {r2_score:}")
