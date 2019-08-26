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
        for detection in result:
            bounding_box = detection['box']

            cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255), 2)

        return image

    def getCropedImages(self, image, result):
        faces_id = {}

        face_count = 0
        for detection in result:
            bounding_box = detection['box']
            cropped = image[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]

            faces_id['face_{}'.format(face_count)] = cropped
            face_count+=1

        return faces_id

