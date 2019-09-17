## USAGE: 

## For downloading data: python data_download.py -d datasets/Face_Recognition.json
## For annotations: python data_download.py -d datasets/Face_Recognition.json -a True

from jsonparser.jsonParser import JSONparser
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="path to JSON data")
ap.add_argument("-a", "--annt", required=False,
	help="create annotations file")
args = vars(ap.parse_args())

parser = JSONparser(args["data"])

data_dict = parser.loadJSONfile()

if args['annt']:
	parser.getAnnotation(data_dict)
	print("[INFO] Done")
else:
	image_urls, image_names = parser.getImageURL(data_dict)
	parser.downloadImage(image_urls, image_names)
	print("[INFO] Done")
