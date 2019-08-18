import json
import requests

## Load dataset
def loadData(dataPath):
    print("[INFO] Loading JSON file.......")
    with open(dataPath, encoding='utf8') as json_file:
        data_dict = json.loads(json_file.read())
    print("[INFO] Loaded")
    return data_dict

def downloadImage(image_urls, image_names):
    for index in range(len(image_urls)):
        print("[INFO] Downloading {}".format(image_names[index]))
        img_data = requests.get(image_urls[index]).content
        with open('images/'+image_names[index], 'wb') as handler:
            handler.write(img_data)


def parseData(data_dict):
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
        
dataPath = 'datasets/Face_Recognition.json'
data_dict = loadData(dataPath)
image_urls, image_names = parseData(data_dict)
downloadImage(image_urls, image_names)


