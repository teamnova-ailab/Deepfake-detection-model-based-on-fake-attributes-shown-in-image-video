import cv2
import threading
import numpy as np
import filter_def

from utils import visualization_utils_face
from utils import visualization_utils_face_part
from utils import visualization_utils_fake

'''
스레드로 각 특징별 탐지 모델 동작시키고 결과값을 배열에 담아서 반환 (탐지 모델 동작에서 시각화 및 시각화에 필요한 부분은 visualization 에 있음)
 - 페이스북 이외 데이터에서 나타나는 특징
'''

# 얼굴 찾는 클래스 -> 얼굴 탐지 후 얼굴에서 나타나는 특징 찾는 모델, 눈코입 찾는 모델 동작
def detect_face(result_rnn, result_face,  frame, sess, detection_graph,
                eye_sess, eye_detection_graph,
                nose_sess, nose_detection_graph,
                mouth_sess, mouth_detection_graph,
                f_d_sess, f_d_detection_graph,
                f_g_sess, f_g_detection_graph,
                f_blur_sess, f_blur_detection_graph,
                e_d_sess, e_d_detection_graph,
                e_g_sess, e_g_detection_graph,
                n_d_sess, n_d_detection_graph,
                n_g_sess, n_g_detection_graph,
                m_d_sess, m_d_detection_graph,
                m_g_sess, m_g_detection_graph):

    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    left, right, top, bottom, label_str = visualization_utils_face.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        use_normalized_coordinates=True,
        line_thickness=4)

    # if (label_str != ""):
    #     # print(img_name, '/', label_str)
    #     result_rnn.append(label_str)
    # else:
    #     # print(img_name, '/', part, '0%,')
    #     result_rnn.append('face' + ' 0%,')

    image = frame[int(top):int(bottom), int(left):int(right)]

    # h, w, channel = image.shape
    # print('h,w', h, w)

    height = bottom - top
    width = right - left
    # h = round(height)
    w = round(width)
    # print('height, width', height, width)

    if left != 0 and height > 0 and width > 0:
        # 눈코입 찾는 스레드
        th_eye = threading.Thread(target=detect_eye,
                                  args=(result_rnn,'eye', w, image, eye_sess, eye_detection_graph,
                                        e_d_sess, e_d_detection_graph,
                                        e_g_sess, e_g_detection_graph))

        th_nose = threading.Thread(target=detect_nose,
                                   args=(result_rnn, result_face, 'nose', w, image, nose_sess, nose_detection_graph,
                                         n_d_sess, n_d_detection_graph,
                                         n_g_sess, n_g_detection_graph))

        th_mouth = threading.Thread(target=detect_mouth,
                                    args=(result_rnn,'mouth', w, image, mouth_sess, mouth_detection_graph,
                                          m_d_sess, m_d_detection_graph,
                                          m_g_sess, m_g_detection_graph))
        th_eye.start()
        th_eye.join()
        th_nose.start()
        th_nose.join()
        th_mouth.start()
        th_mouth.join()

        part_list = []
        th_face_dotnoise = threading.Thread(target=detect_fake,
                                        args=(result_rnn, 'face_dotnoise', image, part_list, f_d_sess, f_d_detection_graph))

        th_face_gridnoise = threading.Thread(target=detect_fake,
                                        args=(result_rnn, 'face_gridnoise', image, part_list, f_g_sess, f_g_detection_graph))
        th_face_blur = threading.Thread(target=detect_fake,
                                        args=(result_rnn, 'face_notblur', image, part_list, f_blur_sess, f_blur_detection_graph))

        th_face_dotnoise.start()
        th_face_dotnoise.join()
        th_face_gridnoise.start()
        th_face_gridnoise.join()
        th_face_blur.start()
        th_face_blur.join()

# 눈 찾는 클래스 -> 눈 찾는 모델 동작 후 눈과 관련된 가짜 특징 찾는 모델 동작
def detect_eye(result_rnn,part, w, image, sess, detection_graph,
               e_d_sess, e_d_detection_graph,
               e_g_sess, e_g_detection_graph) :
    # total = 0
    f_image = filter_def.eye(image)

    label_str, part_list, face_str = face_part_model_run(f_image, part, w, sess, detection_graph)

    # if (label_str != ""):
    #     # print(img_name, '/', label_str)
    #     result_rnn.append(label_str)
    # else:
    #     # print(img_name, '/', part, '0%,')
    #     result_rnn.append(part + ' 0%,')

    th_eye_dotnoise = threading.Thread(target=detect_fake,
                              args=(result_rnn,'eye_dotnoise', image, part_list, e_d_sess, e_d_detection_graph))
    th_eye_gridnoise = threading.Thread(target=detect_fake,
                                        args=(result_rnn, 'eye_gridnoise', image, part_list, e_g_sess,
                                              e_g_detection_graph))

    th_eye_gridnoise.start()
    th_eye_gridnoise.join()
    th_eye_dotnoise.start()
    th_eye_dotnoise.join()

# 코 찾는 클래스 -> 코 찾는 모델 동작 후 코와 관련된 가짜 특징 찾는 모델 동작
def detect_nose(result_rnn, result_face, part, w, image, sess, detection_graph,
                n_d_sess, n_d_detection_graph,
                n_g_sess, n_g_detection_graph):
    # total = 0
    f_image = filter_def.nose(image)

    label_str, part_list, face_str = face_part_model_run(f_image, part, w, sess, detection_graph)

    # if (label_str != ""):
    #     # print(img_name, '/', label_str)
    #     result_rnn.append(label_str)
    # else:
    #     # print(img_name, '/', part, '0%,')
    #     result_rnn.append(part + ' 0%,')
    # # print(part_list)
    result_face.append(face_str)

    th_nose_dotnoise = threading.Thread(target=detect_fake,
                              args=(result_rnn,'nose_dotnoise', image, part_list, n_d_sess, n_d_detection_graph))
    th_nose_gridnoise = threading.Thread(target=detect_fake,
                                         args=(result_rnn, 'nose_gridnoise', image, part_list, n_g_sess,
                                               n_g_detection_graph))

    th_nose_dotnoise.start()
    th_nose_dotnoise.join()
    th_nose_gridnoise.start()
    th_nose_gridnoise.join()


# 입 찾는 클래스 -> 입 찾는 모델 동작 후 입과 관련된 가짜 특징 찾는 모델 동작
def detect_mouth(result_rnn,part, w, image, sess, detection_graph,
                m_d_sess, m_d_detection_graph,
                m_g_sess, m_g_detection_graph):

    f_image = filter_def.mouth(image)
    label_str, part_list, face_str = face_part_model_run(f_image, part, w, sess, detection_graph)

    # if (label_str != ""):
    #     # print(img_name, '/', label_str)
    #     result_rnn.append(label_str)
    # else:
    #     # print(img_name, '/', part, '0%,')
    #     result_rnn.append(part + ' 0%,')

    th_mouth_dotnoise = threading.Thread(target=detect_fake,
                              args=(result_rnn,'mouth_dotnoise', image, part_list, m_d_sess, m_d_detection_graph))
    th_mouth_gridnoise = threading.Thread(target=detect_fake,
                                          args=(result_rnn, 'mouth_gridnoise', image, part_list, m_g_sess,
                                                m_g_detection_graph))
    th_mouth_dotnoise.start()
    th_mouth_dotnoise.join()
    th_mouth_gridnoise.start()
    th_mouth_gridnoise.join()

#각 가짜 특징에 맞는 필터를 적용, 각 특징별 탐지 모델 동작시키는 클래스
def detect_fake(result_rnn, part, image, part_list, sess, detection_graph):
    # print(part)
    if 'face' in part:
        if 'grid' in part:
            f_image = filter_def.face_gridnoise(image)
        elif 'dot' in part:
            # print('dot',image)
            f_image = filter_def.face_dotnoise(image)
        elif 'blur' in part :
            f_image = filter_def.face_blur(image)
    elif 'eye' in part:
        if 'grid' in part:
            f_image = filter_def.eye_gridnoise(image)
        elif 'dot' in part:
            f_image = filter_def.eye_dotnoise(image)
    elif 'nose' in part:
        if 'noise' in part:
            f_image = filter_def.nose_noise(image)
    elif 'mouth' in part:
        if 'noise' in part:
            f_image = filter_def.mouth_noise(image)

    label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)

    if(label_str != ""):
       # print(label_str)
        result_rnn.append(label_str)
    else :
        # print(part, '0%,')
        result_rnn.append(part + ' 0%,')

# 눈,코,입 탐지하는 클래스
def face_part_model_run (f_image, part, w, sess, detection_graph) :
    # 크롭, 필터까지 다 거친 이미지 배열
    image_np_expanded = np.expand_dims(f_image, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Perform the actual detection by running the model with the image as input
    # 이미지를 입력으로 하여 모델을 실행하여 실제 감지 수행
    # 여기서 모델을 거친다
    # inference_start = time.time()
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        # feed_dict = {image_tensor: frame_expanded})
        feed_dict={image_tensor: image_np_expanded})

    label_str, xmin, xmax, ymin, ymax, face_str = \
        visualization_utils_face_part.visualize_boxes_and_labels_on_image_array(
        f_image,
        w,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        part,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.3)
    part_list = [xmin, xmax, ymin, ymax]
    return label_str, part_list, face_str

# 가짜특징 탐지하는 클래스
def fake_model_run (f_image, image, part, part_list, sess, detection_graph) :
    # 크롭, 필터까지 다 거친 이미지 배열
    image_np_expanded = np.expand_dims(f_image, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Perform the actual detection by running the model with the image as input
    # 이미지를 입력으로 하여 모델을 실행하여 실제 감지 수행
    # 여기서 모델을 거친다
    # inference_start = time.time()
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        # feed_dict = {image_tensor: frame_expanded})
        feed_dict={image_tensor: image_np_expanded})

    label_str = visualization_utils_fake.visualize_boxes_and_labels_on_image_array(
        image,
        part,
        part_list,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.1)

    return label_str
