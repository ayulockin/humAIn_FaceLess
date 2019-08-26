## USAGE: python data_download.py -d datasets/Face_Recognition.json

from jsonparser.jsonParser import JSONparser
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="path to JSON data")
ap.add_argument("-n", "--json", required=False,
	help="create new JSON file for validation")
args = vars(ap.parse_args())

parser = JSONparser(args["data"])

data_dict = parser.loadJSONfile()

if args['json']:
	parser.dumpCleanJSON(data_dict)
	print("[INFO] Done")
else:
	image_urls, image_names = parser.getImageURL(data_dict)
	parser.downloadImage(image_urls, image_names)
	print("[INFO] Done")
