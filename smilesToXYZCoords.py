#import necessary libraries
import sys
import subprocess

#function to install missing packages directly from Python
def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


#mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

#import necessary libraries
import os
import glob
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

#extract SMILES strings from all JSON files in the folder
def extract_smiles_from_json(json_file_path):
    smiles_list = []
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    #handle both list or single JSON objects
    if isinstance(data, list):
        items = data
    else:
        items = [data]

    #loop through entries and extract SMILES under 'identifiers'
    for entry in items:
        inputs = entry.get('inputs', {})
        for solution_name, solution_data in inputs.items():
            if isinstance(solution_data, dict) and 'components' in solution_data:
                for component in solution_data['components']:
                    for identifier in component.get('identifiers', []):
                        if identifier.get('type') == 'SMILES':
                            smiles = identifier.get('value')
                            if smiles and smiles.strip():
                                smiles_list.append(smiles.strip())
    return smiles_list

#convert SMILES to XYZ coordinates (3D structure only)
def smiles_to_xyz_coords_only(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = Chem.AddHs(mol)
    try:
        #generate 3D coordinates
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0:
            return ""
        #optimize geometry
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        xyz_lines = []
        #extract atom symbols and positions
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            xyz_lines.append(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}")
        return "\n".join(xyz_lines)
    except Exception as e:
        return ""

#loop through all JSONs, extract SMILES + convert to XYZ, save to CSV
def process_all_json_files_to_csv(path_to_dir):
    json_files = glob.glob(os.path.join(path_to_dir, "*.json"))
    data = []
    for json_file in json_files:
        smiles_list = extract_smiles_from_json(json_file)
        for smiles in smiles_list:
            xyz = smiles_to_xyz_coords_only(smiles)
            if xyz:
                data.append({"smiles": smiles, "xyz": xyz})

    #skip if no valid data found
    if not data:
        return

    #save all results into a single CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(path_to_dir, "smiles_xyz.csv")
    df.to_csv(csv_path, index=False)
    print("done")

#main execution
if __name__ == "__main__":
    pathToDir = "/content/drive/MyDrive/ordData"
    process_all_json_files_to_csv(pathToDir)
