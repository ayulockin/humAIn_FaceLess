import pandas as pd
import os

faces_df = pd.read_csv("datasets/faces.csv")

columns = faces_df.columns

face_ids = faces_df[columns[0]].values

images = os.listdir("face_images")

count_images = 0
faces = []
for image in images:
        if image.endswith('.jpeg'):
                faces.append(image.split('.')[0])
                count_images+=1

		
not_data = [item for item in faces if item not in face_ids]

for item in not_data:
        os.remove('face_images/'+item+'.jpeg')
        print("[INFO] {} removed".format(item+'.jpeg'))
