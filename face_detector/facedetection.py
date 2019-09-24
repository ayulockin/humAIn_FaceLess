import cv2
import numpy as np
from face_detector.mtcnn import MTCNN
from face_detector.utils.stagestatus import StageStatus


class faceDetection():
    def __init__(self, min_face_size: int=20, scale_factor: float=0.709):
        '''
        param min_face_size: minimum size of the face to detect
        '''
        self.__min_face_size = min_face_size
        self.__scale_factor = scale_factor

        self.detector = MTCNN()
        self.__stage1, self.__stage2, self.__stage3 = self.detector.stage1, self.detector.stage2, self.detector.stage3


    @property
    def min_face_size(self):
        return self.__min_face_size

    @min_face_size.setter
    def min_face_size(self, mfc=20):
        try:
            self.__min_face_size = int(mfc)
        except ValueError:
            self.__min_face_size = 20

    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self.__scale_factor, factor_count)]
            min_layer = min_layer * self.__scale_factor
            factor_count += 1

        return scales


    # Run inference through MTCNN to get bounding boxes for detected faces. 
    def detect_faces(self, img) -> list:
        """
        Detects bounding boxes from the specified image.
        :param img: image to process
        :return: list containing all the bounding boxes detected with their keypoints.
        """
        if img is None or not hasattr(img, "shape"):
            raise Exception("Image not valid.")

        height, width, _ = img.shape
        stage_status = StageStatus(width=width, height=height)

        m = 12 / self.__min_face_size
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)

        stages = [self.__stage1, self.__stage2, self.__stage3]
        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in stages:
            result = stage(img, result[0], result[1])

        [total_boxes, points] = result

        bounding_boxes = []

        for bounding_box, keypoints in zip(total_boxes, points.T):

            bounding_boxes.append({
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
                    'confidence': bounding_box[-1]
                }
            )

        return bounding_boxes

    def drawBoundingBox(self, image, result):
        face_count = 0
        for detection in result:
            bounding_box = detection['box']

        
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]

            x, y, w, h = self.getModifyBBox(x,y,w,h,100)

            cv2.rectangle(image,
                      (x, y),
                      (x+w, y+h),
                      (0,155,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,'{}'.format(face_count),(x,y), font, 0.5,(0,20,200),2,cv2.LINE_AA)

            face_count+=1

        return image

    # Get cropped images of the face for further predictions. 
    def getCropedImages(self, image, result):
        faces_id = {}

        face_count = 0
        for detection in result:
            print("[INFO] Face ID", face_count)
            bounding_box = detection['box']

            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]

            x, y, w, h = self.getModifyBBox(x,y,w,h,200)
            print(x,y,w,h)
            if x<0: x=0
            if y<0: y=0

            cropped = image[y:y+h, x:x+w]
            # print(cropped)

            faces_id['face_{}'.format(face_count)] = cropped
            face_count+=1

        return faces_id

    # Modify the bounding box to determine 
    def getModifyBBox(self, x, y, w, h, ratio):

        oldarea = h*w
        newarea = oldarea+(oldarea*(ratio/100))
        # print(newarea)

        hw = int(np.sqrt(newarea))
        # print(hw)
        x = int(x-(hw-w)/2)
        y = int(y-(hw-h)/2)
        # print(hw, hw, x, y)


        return x,y,hw,hw
