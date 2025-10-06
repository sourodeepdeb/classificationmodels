import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Set your directory
pathToDir = "/content/drive/MyDrive/ordMLFiles"
if not os.path.exists(pathToDir):
    print(f"{pathToDir} is not working")
else:
    os.chdir(pathToDir)
    print("Current directory:", os.getcwd())
    print("Files in directory:", os.listdir('.'))

def smiles_to_optimized_xyz(smiles):
    """Convert SMILES to optimized XYZ coordinates (as a single string)."""
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    mol = Chem.MolFromSmiles(smiles, ps)
    if mol is None:
        return ""
    mol = Chem.AddHs(mol)
    try:
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res == -1:
            print(f"Embedding failed for SMILES: {smiles}")
            return ""
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"Optimization failed for SMILES: {smiles}, error: {e}")
            return ""
        conf = mol.GetConformer()
        xyz_lines = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            xyz_lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
        return "\n".join(xyz_lines)
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, error: {e}")
        return ""

# Get all CSV files except those already processed
csv_files = [f for f in os.listdir() if f.endswith('.csv') and '_with_optXYZ.csv' not in f]

for input_csv_filename in csv_files:
    output_csv_filename = input_csv_filename.replace('.csv', '_with_optXYZ.csv')
    print(f"\nProcessing: {input_csv_filename} -> {output_csv_filename}")

    df = pd.read_csv(input_csv_filename)
    if 'COOH SMILES' not in df.columns or 'Amine SMILES' not in df.columns:
        print(f"Skipping {input_csv_filename}: COOH SMILES or Amine SMILES column not found.")
        print("Available columns:", df.columns)
        continue

    # Remove rows where either COOH SMILES or Amine SMILES is missing or empty
    df = df.dropna(subset=['COOH SMILES', 'Amine SMILES'])
    df = df[df['COOH SMILES'].astype(str).str.strip() != '']
    df = df[df['Amine SMILES'].astype(str).str.strip() != '']

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    # Add new columns for optimized XYZ
    df['COOH optXYZ'] = ""
    df['AMINE optXYZ'] = ""

    for idx, row in df.iterrows():
        cooh_smiles = str(row['COOH SMILES']).strip()
        amine_smiles = str(row['Amine SMILES']).strip()
        if cooh_smiles:
            df.at[idx, 'COOH optXYZ'] = smiles_to_optimized_xyz(cooh_smiles)
        if amine_smiles:
            df.at[idx, 'AMINE optXYZ'] = smiles_to_optimized_xyz(amine_smiles)

    df.to_csv(output_csv_filename, index=False)
    print(f"Saved: {output_csv_filename}")

print("done")
