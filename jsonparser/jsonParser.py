import os
import json
import requests
import pandas as pd

## Load dataset

class JSONparser():
    def __init__(self, dataPath):
        self._dataPath = dataPath

    def loadJSONfile(self):
        print("[INFO] Loading JSON file.......")
        with open(self._dataPath, encoding='utf8') as json_file:
            data_dict = json.loads(json_file.read())
        print("[INFO] Loaded")
        return data_dict

    def getImageURL(self, data_dict):
        image_urls = []
        image_names = []
        print("[INFO] Parsing data....")
        for data in data_dict['data']:
            image_url = data['content']
            image_name = image_url[image_url.rfind('__')+2:]
            image_urls.append(image_url)
            image_names.append(image_name)
        print("[INFO] Parsing done")   
        return image_urls, image_names

    def downloadImage(self, image_urls, image_names):
        try:
            print("[INFO] images dir added!")
            os.mkdir('images')
        except:
            print("[NOTICE] images dir already present.")

        for index in range(len(image_urls)):
            print("[INFO] Downloading {}".format(image_names[index]))
            img_data = requests.get(image_urls[index]).content
            with open('images/'+image_names[index], 'wb') as handler:
                handler.write(img_data)

    def dumpCleanJSON(self, data_dict):
        validationJSON = {'data': {}}
        print("[INFO] Preparing Data....")
        for data in data_dict['data']:
            temp_data = {}
            image_url = data['content']
            image_name = image_url[image_url.rfind('__')+2:]
            annotation = data['annotation']
            validationJSON['data'][image_name] = annotation
        try:
            with open("datasets/annotations.json", "w") as f:
                json.dump(validationJSON, f)
        except:
            print("[ERROR] Annotation not created")
