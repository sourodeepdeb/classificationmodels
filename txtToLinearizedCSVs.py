#import necessary libraries
import os
import csv
import re
import time
import pubchempy as pcp

#load directory
pathToDir = "/content/drive/My Drive/ORD_FINAL_CLEAN_3_12"
file_name = "file6_extractedData.txt"

# check if the provided directory exists
if not os.path.exists(pathToDir):
    print(f"{pathToDir} is not working")
else:
    os.chdir(pathToDir)

def extract_names(data_section):
    names = []
    if data_section:
        identifiers = re.findall(r"'type': 'NAME', 'value': '(.*?)'", data_section)
        names = identifiers
    return names


#manual handling of compound abbreviations
def get_smiles_from_pubchem(compound_name, retries=3, delay=2):
    if compound_name.upper() == "DCM":
        return "C(Cl)Cl"
    if compound_name.upper() == "TEA":
        return "CCN(CC)CC"
    if compound_name.upper() == "BOP":
        return "CN(C)P(OC1CN2CNCC2N1)C(N)C"

    attempt = 0
    #check PubChem for compound names
    while attempt < retries:
        try:
            compound = pcp.get_compounds(compound_name, 'name')
            if compound:
                return compound[0].isomeric_smiles
            return "N/A"
        except Exception as e:
            attempt += 1
            if attempt < retries:
                time.sleep(5)
            else:
                return f"Error: {e}"

#list to store reaction data + counter for progress
reaction_data_list = []
reaction_counter = 0

#split the file into individual reactions, separating by using reaction number
with open(file_name, 'r') as file:
    file_content = file.read()

reactions = re.split(r"rxn \d+:", file_content)

#define the headers of the csvs
header = ["Reaction ID", "Yield", "Temperature", "Time"]

#loop through each reaction section to extract relevant data
for reaction in reactions:
    if not reaction.strip():
        continue

    reaction_data = {}

    #extract reaction parameters + names + solvents
    reaction_data["Reaction ID"] = re.search(r"Reaction ID: (.*)", reaction).group(1) if re.search(r"Reaction ID: (.*)", reaction) else "N/A"
    reaction_data["Yield"] = re.search(r"Yield: (.*)", reaction).group(1) if re.search(r"Yield: (.*)", reaction) else "N/A"
    reaction_data["Temperature"] = re.search(r"Temperature: (.*)", reaction).group(1) if re.search(r"Temperature: (.*)", reaction) else "N/A"
    reaction_data["Time"] = re.search(r"Time: (.*)", reaction).group(1) if re.search(r"Time: (.*)", reaction) else "N/A"

    reactant_section = re.search(r"Reactants:\s*(.*?)(?=Solvents:|$)", reaction, re.DOTALL)
    reactant_names = extract_names(reactant_section.group(1) if reactant_section else None)

    solvent_section = re.search(r"Solvents:\s*(.*?)(?=Reactants:|$)", reaction, re.DOTALL)
    solvent_names = extract_names(solvent_section.group(1) if solvent_section else None)

    #add reactant and solvent columns to csv header
    for i in range(1, len(reactant_names) + 1):
        header.append(f"Reactant {i} Name")
        header.append(f"Reactant {i} SMILES")

    for i in range(1, len(solvent_names) + 1):
        header.append(f"Solvent {i} Name")
        header.append(f"Solvent {i} SMILES")

    #create base row for reaction data
    row = [
        reaction_data["Reaction ID"],
        reaction_data["Yield"],
        reaction_data["Temperature"],
        reaction_data["Time"]
    ]

    #append reactant names and SMILES strings
    for name in reactant_names:
        row.append(name)
        row.append(get_smiles_from_pubchem(name))

    #append solvent names and SMILES strings
    for name in solvent_names:
        row.append(name)
        row.append(get_smiles_from_pubchem(name))

    #add reaction's data to overall list
    reaction_data_list.append(row)

    reaction_counter += 1

  #to avoid issues with rate limiting/overwhelming the API, pause every 50 reactions
    if reaction_counter % 50 == 0:
        print(f"{reaction_counter} reactions ran. Pausing for 30 second")
        time.sleep(30)

csv_filename = 'V2_File6.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(reaction_data_list)
