# humAIn_FaceLess

## Guide to Setup

Since git LFS wasn't used huge files weren't provided along with this repository. Thus follow these steps to access all the functionality provided with this repository. 

### Download Images

1. Clone this repository to you local system.

2. The dataset provided by the HumAIn team is in `JSON` format. The same can be found in the `datasets` dir. I have added a parent key named `data`so to the provided data. If you have a test dataset in `JSON` and want to use the same for evaluation make sure to add this parent key.

3. To download images open `command prompt` or `terminal`in the root of the cloned repository. 

   ```bash
   python data_download.py -d datasets/Face_Recognition.json
   ```

### Download Weights

Download my trained model `weight files` from [here]() [ADD LINK HERE] . The files should be in this relative path: `face_detector\weights`

### Download Additional Dataset (Optional)

1. In order to train my model for Age, Gender and Ethnicity prediction, I have used [UTKFace](https://susanqq.github.io/UTKFace/) dataset along side the data provided.

2. In order to train my model for Emotion prediction I have used the dataset from [this](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data ) Kaggle competition.

   You can run `prepareUTKFaceData.py`(Optional)

   ```bash
   python prepareUTKFaceData.py -d path_to_downloaded_UTKFace_dataset
   ```

   This will create a `face_dataset.csv` dataset along with a directory in the root named `UTKFaceimages`. The `face_dataset.csv` contains `image_id, age, gender, ethnicity` as input image name along with class label. The `UTKFaceimages` directory contain images with `image_id`. This renaming of file is done as a part of data preprocessing which makes it easier while preparing data for training. 





