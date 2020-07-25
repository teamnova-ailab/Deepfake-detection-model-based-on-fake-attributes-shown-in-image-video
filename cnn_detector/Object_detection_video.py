######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
# from fake_filter import filter_def , mtcnn_crop_def , opencv_detect_def

# 코드 시작시간
code_start = time.time()

config= tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session= tf.compat.v1.Session(config=config)

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'models'
LABELMAP_NAME = 'labelmap'


# =========================================================================

# 폴더 내 이미지 불러오기
folder_path = ""

folder_list = os.listdir(folder_path)

# 결과 이미지 저장 폴더
output_dir = ""

# 모델, 라벨맵 이름 설정
model = 'f_blur_faster.pb'
label = 'f_blur_faster.pbtxt'

# 출력되는 박스 percentage
percent = 0.1

# =========================================================================
# Grab path to current working directory
CWD_PATH = os.getcwd()

# 모델 불러오는 부분 => 현재 단일 모델 불러옴
# Path to frozen detection graph .pb file, which contains the model that is used for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,model)
#print("model_path : "+PATH_TO_CKPT)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME,label)
#print("label_path : "+PATH_TO_LABELS)

# # Number of classes the object detector can identify
# max_num_classes 를 지정해주는 값
# 최대 라벨 개수로 지정해주면 될 것 같음
# label class 개수보다 작은 값이 들어가면 N/A 라고 나옴
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

count=0
print("filename, img_input_time, process_time, label, percent, label, percent,")
for item in folder_list:  # 폴더의 파일이름 얻기




    PATH_TO_VIDEO = folder_path+item

    cap = cv2.VideoCapture(PATH_TO_VIDEO)

    start = time.time()

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    #코드 실행 시작시간
    start_time=time.time() - start

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value

    frame_num = 300
    label_add = ''

    while frame_num:

        ret, frame = cap.read()
        # face_result = list()
        if ret == 0:
            break
        if (int(cap.get(1)) % num == count):
            img_num = int(cap.get(1))




            image = cv2.imread(frame)
            h, w, channel = image.shape

            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            label_str=""
            image, label_str= vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=percent)
            label_str = label_str

            # 탐지한 라벨만 출력
            if (label_str != ""):

                count = count + 1
            print(item+", "+ str(start_time)+", "+str(time.time() - start)+", "+label_str)


        # 잘 탐지했는지 확인

        # cv2.imshow('Object detector', frame)
        # Press 'q' to quit
        # if cv2.waitKey(1) == ord('q'):
        #    break
        frame_num -= 1


print("code finish time :", time.time() - code_start)  # 현재시각 - 시작시간 = 실행 시간
# Clean up

cv2.destroyAllWindows()
