import os
import pandas as pd
import string
import random
from shutil import copyfile

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="path to downloaded UTKFace Images")
args = vars(ap.parse_args())

field_names = ['image_id', 'age', 'gender', 'ethnicity']
face_dataset = pd.DataFrame(columns = field_names)

scr_image_path = args["data"]
images = os.listdir(scr_image_path)

try:
	os.makedirs('UTKFaceimages')
	print("images dir made")
except:
	print("dir already present")

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

c = 0
for image in images:
	print("[INFO] Renaming: ", image)
	image_id = randomString()
	age, gender, ethnicity = image.split('_')[0:3]
	face_dataset.loc[c] = [image_id, age, gender, ethnicity]
	copyfile(scr_image_path+image, 'UTKFaceimages/'+image_id+'.jpg')

	c+=1

print("[INFO] csv dataset preparing....")
face_dataset.to_csv('datasets/face_dataset.csv', index=False)
print("[INFO] Job done")
