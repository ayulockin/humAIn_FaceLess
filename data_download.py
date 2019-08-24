from jsonparser.jsonParser import JSONparser
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="path to JSON data")
args = vars(ap.parse_args())

parser = JSONparser(args["data"])

data_dict = parser.loadJSONfile()
image_urls, image_names = parser.getImageURL(data_dict)

parser.downloadImage(image_urls, image_names)