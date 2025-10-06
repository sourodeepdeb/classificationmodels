#import necessary libraries
import os
import json
from google.colab import drive
from ord_schema.message_helpers import load_message
from ord_schema.proto import dataset_pb2
from google.protobuf.json_format import MessageToJson

#connect to google drive directory where files are stored
drive.mount('/content/gdrive/ordData')


#pb.gz --> .json conversion
def ordToJsonFiles(inpFile, outFile):
    try:
      #load the dataset
        dataset = load_message(inpFile, dataset_pb2.Dataset)

        #empty list which will store all of the .json data for each reaction + convert each reaction object to a json dictionary
        jsonFileExtractedData = []
        for rxn in dataset.reactions:
            jsonFileRxns = json.loads(
                MessageToJson(
                    message=rxn,
                    including_default_value_fields=False,
                    preserving_proto_field_name=True,
                    indent=2,
                    sort_keys=False,
                    use_integers_for_enums=False,
                    descriptor_pool=None,
                    float_precision=None,
                    ensure_ascii=True,
                )
            )
            #add each reaction json to the list
            jsonFileExtractedData.append(jsonFileRxns)

        #add the outputs to a .json file
        with open(outFile, 'w') as f:
            json.dump(jsonFileExtractedData, f, indent=2)
        print(f"{inpFile} --> {outFile}")

pathToDir = "/content/drive/MyDrive/ordData"

#loop through all files in directory to convert any files ending in pb.gz to .json
for filename in os.listdir(pathToDir):
  if filename.endswith('.pb.gz'):
    inpFile = os.path.join(pathToDir, filename)
    outFile = os.path.join(pathToDir, f"{os.path.splitext(filename)[0]}.json")
    ordToJsonFiles(inpFile, outFile)

print("done")
