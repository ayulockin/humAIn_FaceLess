import cv2
import tqdm
import os
import pandas as pd
import string
import random
from jsonparser.jsonParser import JSONparser

try:
    os.mkdir('face_images')
except:
    print("[INFO] Dir already present")

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def getLabels(face_dict):
    labels = image['label']
    
    for label in labels:
        if label.split('_')[0]=='Emotion':
            emotion = label
        elif label.split('_')[0]=='E':
            race = label
        elif label.split('_')[0]=='G':
            gender = label
        elif label.split('_')[0]=='Age':
            age = label
        elif label == "Not_Face":
            return "Not_Face"    
    try:
        emotion
    except NameError:
        emotion = None
    try:
        gender
    except NameError:
        gender = None
    try:
        race
    except NameError:
        race = None
    try:
        age
    except NameError:
        age = None
    
    return [image_id, gender, race, age, emotion]

def getCoordinates(face_dict):
    img_W = image['imageWidth']
    img_H = image['imageHeight']
    
    points = image['points']
    box = []

    for point in points:
        for key, val in point.items():
            if key=='x':
                box.append(int(round(val*img_W)))
            else:
                box.append(int(round(val*img_H)))

    return box

def makeFaces(image, image_id, bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    
    cropped = image[y:h, x:w]
    cv2.imwrite("face_images/"+image_id+".jpeg", cropped)

parser = JSONparser("datasets/annotations.json")

data_dict = parser.loadJSONfile()

field_name = ['face_id', 'gender', 'race', 'age', 'emotion']
faces = pd.DataFrame(columns=field_name)

c = 0
for data in data_dict['data']:
    if data.split('.')[-1] != 'gif':
        img = cv2.imread('images/'+data)
        print("[INFO] Cropping for image: {}".format(data))
        image_data = data_dict['data'][data]
        for image in image_data:
            image_id = randomString()

            labels = getLabels(image)
            if labels != "Not_Face":
                faces.loc[c] = labels
                c+=1

            bbox = getCoordinates(image)

            makeFaces(img, image_id, bbox)

print("[INFO] csv dataset preparing....")
faces.to_csv('datasets/faces.csv', index=False)
print("[INFO] Job done")



