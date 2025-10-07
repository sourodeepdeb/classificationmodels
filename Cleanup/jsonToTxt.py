#import necessary libraries
import os
import json
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors
from google.colab import drive

#mount to google drive which contains all the .json files
drive.mount('/content/drive')

pathToDir = "/content/drive/MyDrive/ordData"

#check if files exist or not (error handling)
if not os.path.exists(pathToDir):
    print(f"{pathToDir} doesn't exist")
else:
    os.chdir(pathToDir)
    def infoExtract(jsonData):
        extracted = []
        #initialize dictionary to hold extracted data for one reaction + parse reaction inputs
        for rxn in jsonData:
            rxnData = {
                'mw': [], 'logP': [], 'smiles': [], 'casSmiles': [],
                'reactants': [], 'reagents': [], 'solvents': [], 'yield': [],
                'temperature': [], 'time': [], 'rxnId': [],
                'productSmiles': [], 'productNames': []
            }
            inputs = rxn.get('inputs', {})
            hasCarboxylicAcid = False
            hasAmine = False
            for componentGroup in inputs.values():
                if isinstance(componentGroup, dict) and 'components' in componentGroup:
                    for item in componentGroup['components']:
                        for identifier in item.get('identifiers', []):
                            if identifier['type'] == 'SMILES':
                                smiles = identifier['value']
                                rxnData['smiles'].append(smiles)

                                #rdkit for computing descriptors + filtering functional groups
                                mol = Chem.MolFromSmiles(smiles)
                                if mol:
                                    rxnData['mw'].append(Descriptors.ExactMolWt(mol))
                                    rxnData['logP'].append(Descriptors.MolLogP(mol))
                                    if '(=O)O' in smiles:
                                        hasCarboxylicAcid = True
                                    if 'N' in smiles:
                                        hasAmine = True

                        #sort components by reaction role
                        role = item.get('reaction_role', '')
                        if role == 'REACTANT':
                            rxnData['reactants'].append(item)
                        elif role == 'REAGENT':
                            rxnData['reagents'].append(item)
                        elif role == 'SOLVENT':
                            rxnData['solvents'].append(item)

            #skip reactions not containing a carboxylic acid or amine
            if not (hasCarboxylicAcid and hasAmine):
                continue
            outcomes = rxn.get('outcomes', [])

            #get product identifiers --> smiles/names
            if outcomes:
                for product in outcomes[0].get('products', []):
                    prodSmiles = None
                    prodName = None
                    for identifier in product.get('identifiers', []):
                        if identifier.get('type') == 'SMILES':
                            prodSmiles = identifier.get('value')
                        if identifier.get('type') == 'NAME':
                            prodName = identifier.get('value')

                    #store valid product information
                    if prodSmiles is not None:
                        rxnData['productSmiles'].append(prodSmiles)
                        rxnData['productNames'].append(prodName if prodName is not None else "")

                    # Stop after first product with yield data
                    for measurement in product.get('measurements', []):
                        if measurement.get('type') == 'YIELD':
                            rxnData['yield'] = measurement.get('percentage', {}).get('value')
                            break
                    if rxnData['yield'] is not None:
                        break
            rxnData['temperature'] = rxn.get('conditions', {}).get('temperature', {}).get('setpoint', {}).get('value')
            if rxnData['temperature'] == 'none':
                rxnData['temperature'] = '23.0'
            rxnData['time'] = rxn.get('outcomes', [{}])[0].get('reaction_time', {}).get('value')
            rxnData['rxnId'] = rxn.get('reaction_id')

            # Append fully extracted reaction data
            extracted.append(rxnData)
        return extracted

    #loops through all .json files in directory --> extracts reaction information and writes the information as a .txt output
    def procJsonFiles(directory):
        outFile = os.path.join(directory, 'extractedData.txt')
        with open(outFile, 'w') as outfile:
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r') as jsonfile:
                        data = json.load(jsonfile)
                        extracted = infoExtract(data)
                        output = f"File: {filename}\n"
                        for i, rxn in enumerate(extracted, 1):
                            output += f"rxn {i}:\n"
                            output += f"  Reaction ID: {rxn['rxnId']}\n"
                            output += f"  MW: {rxn['mw']}\n"
                            output += f"  LogP: {rxn['logP']}\n"
                            output += f"  SMILES: {rxn['smiles']}\n"
                            output += f"  CAS SMILES: {rxn['casSmiles']}\n"
                            output += f"  Reactants: {rxn['reactants']}\n"
                            output += f"  Reagents: {rxn['reagents']}\n"
                            output += f"  Solvents: {rxn['solvents']}\n"
                            output += f"  Product SMILES: {rxn['productSmiles']}\n"
                            output += f"  Product Names: {rxn['productNames']}\n"
                            if rxn['yield'] != 'None':
                                output += f"  Yield: {rxn['yield']}\n"
                            output += f"  Temperature: {rxn['temperature']}\n"
                            output += f"  Time: {rxn['time']}\n\n"

                        #print out the outputs
                        outfile.write(output)
                        print(output)


    #call and run the processing function to convert .json files to .txt
    procJsonFiles(pathToDir)

    #display file text
    print("extractedData.txt:")
    with open(os.path.join(pathToDir, 'extractedData.txt'), 'r') as f:
        print(f.read())
