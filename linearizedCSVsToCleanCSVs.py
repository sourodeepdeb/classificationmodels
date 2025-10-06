#import necessary libraries
import csv
import time
import pubchempy as pcp
from rdkit import Chem
import os
import re
import logging
from rdkit import RDLogger

#disable unnecessary RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')

#load input/output file paths
pathToDir = "/content/drive/My Drive/ORD_FINAL_CLEAN_3_12/CSV_V2_Uncleaned"
input_csv_filename = 'V2_File10.csv'
output_csv_filename = 'V2_File10_Done.csv'

#check if directory exists, otherwise print error
if not os.path.exists(pathToDir):
    print(f"{pathToDir} is not working")
else:
    os.chdir(pathToDir)

#list of solvent SMILES for identification
solvent_list = [
    "CN(C)C=O", "CS(=O)C", "CC(=O)C", "CC#N", "CCCCCC", "C1=CC=CC=C1",
    "CC(C)O", "CCO", "CO", "C(CCl)Cl", "C(Cl)Cl", "CCOCC", "C1CCOCC1",
    "CN1CCCC1=O", "C1=CC=NC=C1", "C1=CC=C(C=C1)Cl", "CCCCO", "C1CCCCC1",
    "CC(=O)O", "CC(C)CC(=O)C", "C1COCCO1", "C(Cl)(Cl)Cl"
]

#list of known coupling agents represented by SMILES
coupeling_agents = [
    "C1CCC(CC1)N=C=NC2CCCCC2", "CC(C)N=C=NC(C)C", "CCN=C=NCCCN(C)C.Cl", "C1=CC=C2C(=C1)N=NN2O",
    "C1CCN(C1)[P+](N2CCCC2)(N3CCCC3)ON4C5=CC=CC=C5N=N4.F[P-](F)(F)(F)(F)F",
    "CN(C)[P+](N(C)C)(N(C)C)ON1C2=CC=CC=C2N=N1.F[P-](F)(F)(F)(F)F",
    "C1CCN(C1)[P+](N2CCCC2)(N3CCCC3)ON4C5=C(C=CC=N5)N=N4.F[P-](F)(F)(F)(F)F",
    "C1CCN(C1)[P+](N2CCCC2)(N3CCCC3)Br.F[P-](F)(F)(F)(F)F",
    "CN(C)[P+](N(C)C)(N(C)C)ON1C2=CC=CC=C2N=N1.[Cl-]",
    "CN(C)P(OC1CN2CNCC2N1)C(N)C",
    "CN(C)C(=[N+](C)C)ON1C2=CC=CC=C2N=N1.F[P-](F)(F)(F)(F)F",
    "[B-](F)(F)(F)F.CN(C)C(=[N+](C)C)ON1C2=CC=CC=C2N=N1",
    "[B-](F)(F)(F)F.CN(C)C(=[N+](C)C)ON1C2=C(C=CC=N2)N=N1",
    "CN(C)C(=[N+](C)C)N1C2=C(N=CC=C2)[N+](=N1)[O-].F[P-](F)(F)(F)(F)F",
    "[B-](F)(F)(F)F.CCOC(=O)C(=NOC(=[N+](C)C)N(C)C)C#N",
    "CN(C)C(=[N+](C)C)ON1C2=C(C=CC(=C2)Cl)N=N1.F[P-](F)(F)(F)(F)F",
    "CCOC(=O)/C(=N\\OC(=[N+](C)C)N1CCOCC1)/C#N.F[P-](F)(F)(F)(F)F",
    "[B-](F)(F)(F)F.CN(C)C(=[N+](C)C)ON1C(=O)C2=CC=CC=C2N=N1",
    "[B-](F)(F)(F)F.CN(C)C(=[N+](C)C)ON1C(=O)CCC1=O",
    "[B-](F)(F)(F)F.CN(C)C(=[N+](C)C)ON1C(=O)C2C3CC(C2C1=O)C=C3",
    "[B-](F)(F)(F)F.CN(C)C(=[N+](C)C)ON1C=CC=CC1=O",
    "CCOP(=O)(OCC)ON1C(=O)C2=CC=CC=C2N=N1",
    "C1=CN(C=N1)C(=O)N2C=CN=C2",
    "CN(C)C(=[N+](C)C)Cl.F[P-](F)(F)(F)(F)F"
]

attempt = 0  #initialize retry counter for PubChem API

#function to retrieve molecular properties from PubChem using SMILES
def get_property_from_pubchem(smiles, property_name, retries=3, delay=2):
    try:
        compound = pcp.get_compounds(smiles, 'smiles')
        if compound:
            compound = compound[0]
            if property_name == "MolecularWeight":
                return compound.molecular_weight
            elif property_name == "XLogP":
                return compound.xlogp
            else:
                return "N/A"
        return "N/A"
    except Exception as e:
        #retry logic if PubChem call fails
        attempt += 1
        if attempt < retries:
            time.sleep(5)
        else:
            return "N/A"

#function to detect presence of amine groups (primary or secondary)
def get_amine(smiles):
    amine_patterns = {"primary_amine": "[NH2]", "secondary_amine": "[NH]"}
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for amine_type, pattern in amine_patterns.items():
            substructure = Chem.MolFromSmarts(pattern)
            if mol.HasSubstructMatch(substructure):
                return True
    return False

#function to detect generic nitrogen-containing additives
def get_additive(smiles):
    amine_patterns = {"additive": "[N]"}
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for amine_type, pattern in amine_patterns.items():
            substructure = Chem.MolFromSmarts(pattern)
            if mol.HasSubstructMatch(substructure):
                return True
    return False


#open input CSV to read reaction data
with open(input_csv_filename, mode='r') as infile:
    reader = csv.reader(infile)
    header = next(reader)

    #extract key column indexes
    reaction_id_index = header.index("Reaction ID")
    yield_index = header.index("Yield")
    time_index = header.index("Time")
    temp_index = header.index("Temperature")

    #define new CSV header for cleaned output
    new_header = ["Reaction ID", "Yield", "Time", "Temperature",
                  "Solvent SMILES", "Solvent",
                  "Coupling Agent SMILES", "Coupling Agent",
                  "COOH SMILES", "COOH", "COOH MW", "COOH logP",
                  "Amine SMILES", "Amine", "Amine MW", "Amine logP",
                  "Additive SMILES", "Additive"]

    new_rows = []

    #iterate through all reaction rows
    for row in reader:
        #extract main reaction parameters
        reaction_id = row[reaction_id_index]
        yield_val = row[yield_index]
        time_val = row[time_index]
        temp_val = row[temp_index]

        #initialize default placeholders
        solvent_found = "N/A"
        solvent_smiles_found = "N/A"
        coupling_agent_found = "N/A"
        coupling_agent_smiles_found = "N/A"
        cooh_found = "N/A"
        cooh_smiles_found = "N/A"
        amine_found = "N/A"
        amine_smiles_found = "N/A"
        additive_found = "N/A"
        additive_smiles_found = "N/A"

        #avoids duplicates
        added_compounds = set()

        #iterate through reactant and solvent pairs
        for i in range(4, len(row), 2):
            reactant_smiles = row[i]
            reactant_name = row[i + 1] if i + 1 < len(row) else ""

            #identify solvent from known solvent list
            if reactant_name not in added_compounds and any(solvent.lower() in reactant_name.lower() for solvent in solvent_list):
                solvent_smiles_found = reactant_smiles
                solvent_found = reactant_name
                added_compounds.add(reactant_name)

            #identify coupling agents from known list
            if reactant_name not in added_compounds and reactant_name != solvent_found:
                matched = False
                for coupling_agent in coupling_agents:
                    if coupling_agent.lower() in reactant_name.lower():
                        coupling_agent_smiles_found = reactant_smiles
                        coupling_agent_found = reactant_name
                        added_compounds.add(reactant_name)
                        matched = True
                        break

            #identify carboxylic acid reactants (COOH group)
            if reactant_name not in added_compounds and reactant_name != solvent_found and "C(=O)O" in reactant_name:
                cooh_smiles_found = reactant_smiles
                cooh_found = reactant_name
                added_compounds.add(reactant_name)

            #identify amine-containing reactants
            if reactant_name not in added_compounds and reactant_name != solvent_found and get_amine(reactant_name):
                amine_smiles_found = reactant_smiles
                amine_found = reactant_name
                added_compounds.add(reactant_name)

            #identify nitrogen-based additives
            if reactant_name not in added_compounds and reactant_name != solvent_found and get_additive(reactant_name):
                additive_smiles_found = reactant_smiles
                additive_found = reactant_name
                added_compounds.add(reactant_name)

        #skip invalid yields
        try:
            yield_val = float(yield_val)
            if yield_val > 100 or yield_val < 0:
                continue
        except ValueError:
            continue

        #skip incomplete reaction records
        if coupling_agent_smiles_found == "N/A" or cooh_smiles_found == "N/A" or amine_smiles_found == "N/A":
            continue

        #extract molecular properties for COOH and Amine reactants
        cooh_mw = get_property_from_pubchem(cooh_found, "MolecularWeight")
        cooh_logp = get_property_from_pubchem(cooh_found, "XLogP")
        amine_mw = get_property_from_pubchem(amine_found, "MolecularWeight")
        amine_logp = get_property_from_pubchem(amine_found, "XLogP")

        #compile all extracted data into one row
        new_row = [
            reaction_id,
            yield_val,
            time_val,
            temp_val,
            solvent_found,
            solvent_smiles_found,
            coupling_agent_found,
            coupling_agent_smiles_found,
            cooh_found,
            cooh_smiles_found,
            cooh_mw,
            cooh_logp,
            amine_found,
            amine_smiles_found,
            amine_mw,
            amine_logp,
            additive_found,
            additive_smiles_found
        ]
        new_rows.append(new_row)

        #to avoid issues with rate limiting/overwhelming the API, pause every 100 rows
        if len(new_rows) % 100 == 0:
            print(f"{len(new_rows)} reactions processed. Pausing for 30 seconds.")
            time.sleep(30)

#write the processed data into a new CSV
with open(output_csv_filename, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(new_header)
    writer.writerows(new_rows)
