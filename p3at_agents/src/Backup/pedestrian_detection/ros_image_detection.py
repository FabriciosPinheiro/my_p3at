#!/usr/bin/env python3
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#for detection
import numpy as np
import os
import imutils

class image_detection:
 
    def __init__(self):
        self.NMS_THRESHOLD=0.3
        self.MIN_CONFIDENCE=0.2

        #Topic for publication
        #self.image_pub = rospy.Publisher("/sim_p3at/cam",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/sim_p3at/camera/image_raw",Image,self.callback)

        self.dir = os.path.dirname(__file__)

        labelsPath = self.dir+"/coco.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")

        weights_path = self.dir+"/yolov4-tiny.weights"
        config_path = self.dir+"/yolov4-tiny.cfg"

        self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        '''
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        '''

        #layer_name = model.getLayerNames()
        #layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]

        layer_indexes = self.model.getUnconnectedOutLayers()
        self.layer_names = [self.model.getLayerNames()[i - 1] for i in layer_indexes.flatten()]

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        results = self.pedestrian_detection(cv_image, self.model, self.layer_names,
                personidz=self.LABELS.index("person"))
        
        image_height, image_width = cv_image.shape[:2]
        center_x = image_width // 2
        center_y = image_height // 2
        rect_width = 50  # Largura do retângulo 
        rect_height = 90  # Altura do retângulo 
        rect_x = center_x - (rect_width // 2)
        rect_y = center_y - (rect_height // 2) - 35
        cv2.rectangle(cv_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 2)

        for res in results:
            cv2.rectangle(cv_image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

            green_x1 = res[1][0]
            green_y1 = res[1][1]
            green_x2 = res[1][2]
            green_y2 = res[1][3]

            print('Green_x1 = ', green_x1)
            print('Green_y1 = ', green_y1)
            print('Green_x2 = ', green_x2)
            print('Green_x2 = ', green_y2)
		
            blue_x1 = rect_x
            blue_y1 = rect_y
            blue_x2 = rect_x + rect_width
            blue_y2 = rect_y + rect_height

            print('X1 = ', blue_x1)
            print('Y1 = ', blue_y1)
            print('X2 = ', blue_x2)
            print('Y2 = ', blue_y2)
		
            diff_left = green_x1 - blue_x1
            diff_right = blue_x2 - green_x2
            diff_top = green_y1 - blue_y1
            diff_bottom = blue_y2 - green_y2
		
            print("Diferença esquerda:", diff_left)
            print("Diferença direita:", diff_right)
            print("Diferença superior:", diff_top)
            print("Diferença inferior:", diff_bottom)

        cv2.imshow("Detection",cv_image)

        key = cv2.waitKey(1)


        #self.cv_image = cv2.resize(self.cv_image, (400, 400), interpolation = cv2.INTER_NEAREST)

        #print(type(self.cv_image))

        #cv2.imshow("Real Image", self.cv_image)
        #cv2.imshow("Raw Image", raw_image)
        #cv2.waitKey(1) 

        #try:
        #    self.image_pub.publish(self.bridge.cv2_to_imgmsg(raw_image, "bgr8"))
        #except CvBridgeError as e:
        #    print(e)

    def pedestrian_detection(self, image, model, layer_names, personidz=0):
        (H, W) = image.shape[:2]
        results = []


        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        model.setInput(blob)
        layerOutputs = model.forward(layer_names)

        boxes = []
        centroids = []
        confidences = []

        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == personidz and confidence > self.MIN_CONFIDENCE:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idzs = cv2.dnn.NMSBoxes(boxes, confidences, self.MIN_CONFIDENCE, self.NMS_THRESHOLD)
        # ensure at least one detection exists
        if len(idzs) > 0:
            # loop over the indexes we are keeping
            for i in idzs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(res)
        # return the list of results
        return results

def main(args):
    ic = image_detection()
    rospy.init_node('image_detection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)