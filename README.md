# humAIn_FaceLess

## Installations

Since git LFS wasn't used huge files weren't provided along with this repository. Thus follow these steps to access all the functionality provided with this repository. 

### Download Images

1. Clone this repository to you local system.

2. The dataset provided by the HumAIn team is in `JSON` format. The same can be found in the `datasets` dir. I have added a parent key named `data`so to the provided data. If you have a test dataset in `JSON` and want to use the same for evaluation make sure to add this parent key.

3. To download images open `command prompt` or `terminal`in the root of the cloned repository. 

   ```bash
   python data_download.py -d datasets/Face_Recognition.json
   ```

4. To create a cleaner `json` file with annotations.

   ```
   python data_download.py -d datasets/Face_Recognition.json -n True
   ```

### Download Weights

Download my trained model `weight files` from [here](https://drive.google.com/drive/folders/15YPWuZU17Tp6oi2i5vdUlxt26HKiKqyp?usp=sharing). The files should be in this relative path: `face_detector\weights`

### Download Additional Dataset (Optional)

1. In order to train my model for Age, Gender and Ethnicity prediction, I have used [UTKFace](https://susanqq.github.io/UTKFace/) dataset along side the data provided.

2. In order to train my model for Emotion prediction I have used the dataset from [this](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data ) Kaggle competition.

   You can run `prepareUTKFaceData.py`(Optional)

   ```bash
   python prepareUTKFaceData.py -d path_to_downloaded_UTKFace_dataset
   ```

   This will create a `face_dataset.csv` dataset along with a directory in the root named `UTKFaceimages`. The `face_dataset.csv` contains `image_id, age, gender, ethnicity` as input image name along with class label. The `UTKFaceimages` directory contain images with `image_id`. This renaming of file is done as a part of data preprocessing which makes it easier while preparing data for training. 

## USAGE

Go to the root of this repository and open command prompt or in your terminal run `main.py` script. You need to specify the path to the test image. 

```bash
python main.py -i examples/example1.jpeg
```

If you want to see the bounding box detected you can run this. 

```bash
python main.py -i examples/example1.jpeg -b True
```

This will do the following:

1. Will import all the packages.
2. Will build the model and load weights. This may take some time in the first time. 
3. With the given input image run the inference through Face Detector. This will output bounding boxes of the detected faces.
4. It will use this output and crop the faces from the input image which will be the input to three parallel detectors. 
5. The result of the detector will be printed on the terminal.



