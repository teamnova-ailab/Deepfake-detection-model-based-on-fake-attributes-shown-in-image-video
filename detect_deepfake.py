import cv2
import threading
import numpy as np
import filter_def

from utils import visualization_utils_face
from utils import visualization_utils_fake

'''
스레드로 각 특징별 탐지 모델 동작시키고 결과값을 배열에 담아서 반환 (탐지 모델 동작에서 시각화 및 시각화에 필요한 부분은 visualization 에 있음)
 - 눈코입에 나타나는 특징을 찾는 스레드
'''

# 얼굴 찾는 클래스 -> 얼굴 탐지 후 가짜특징 찾는 모델 동작
def detect_face_deepfake(result_deepfake, frame, sess, detection_graph,
                            alae_sess, alae_detection_graph,
                            stylegan_sess, stylegan_detection_graph,
                            stargan_sess, stargan_detection_graph,
                            faceswap_sess, faceswap_detection_graph):

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



    th_alae = threading.Thread(target=detect_fake,
                                        args=(result_deepfake, 'alae', image, alae_sess, alae_detection_graph))

    th_stylegan = threading.Thread(target=detect_fake,
                                        args=(result_deepfake, 'stylegan', frame, stylegan_sess, stylegan_detection_graph))
    th_stargan = threading.Thread(target=detect_fake,
                                        args=(result_deepfake, 'stargan', image, stargan_sess, stargan_detection_graph))
    th_faceswap = threading.Thread(target=detect_fake,
                                  args=(result_deepfake, 'faceswap', image, faceswap_sess, faceswap_detection_graph))

    th_began = threading.Thread(target=detect_fake,
                                   args=(result_deepfake, 'began', image, faceswap_sess, faceswap_detection_graph))
    th_mwgan = threading.Thread(target=detect_fake,
                                   args=(result_deepfake, 'mwgan', image, faceswap_sess, faceswap_detection_graph))
    th_casualgan = threading.Thread(target=detect_fake,
                                   args=(result_deepfake, 'casualgan', image, faceswap_sess, faceswap_detection_graph))
    th_hat = threading.Thread(target=detect_fake,
                                   args=(result_deepfake, 'hat', image, faceswap_sess, faceswap_detection_graph))

    th_alae.start()
    th_alae.join()
    th_stylegan.start()
    th_stylegan.join()
    th_stargan.start()
    th_stargan.join()
    th_faceswap.start()
    th_faceswap.join()
    # th_began.start()
    # th_began.join()
    # th_mwgan.start()
    # th_mwgan.join()
    # th_casualgan.start()
    # th_casualgan.join()
    # th_hat.start()
    # th_hat.join()



#각 가짜 특징에 맞는 필터를 적용, 각 특징별 탐지 모델 동작시키는 클래스
def detect_fake(result_deepfake, part, image, sess, detection_graph):
    # print(part)

    
    part_list = []

    
    if 'alae' in part:
        #f_image = image
        f_image = filter_def.alae(image)
        label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)
    elif 'stylegan' in part:
       # f_image = image
        f_image = filter_def.stylegan(image)
        label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)
    elif 'stargan' in part:
       # f_image = image
        f_image = filter_def.stargan(image)
        label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)
    elif 'faceswap' in part:
       # f_image = image
       f_image = filter_def.faceswap(image)
       label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)

    elif 'began' in part:
       # f_image = image
       f_image = filter_def.faceswap(image)
       label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)
    elif 'mwgan' in part:
       # f_image = image
       f_image = filter_def.faceswap(image)
       label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)
    elif 'casualgan' in part:
       # f_image = image
       f_image = filter_def.faceswap(image)
       label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)
    elif 'hat' in part:
       # f_image = image
       f_image = filter_def.faceswap(image)
       label_str = fake_model_run(f_image, image, part,  part_list, sess, detection_graph)
   

    

    if(label_str != ""):
       # print(label_str)
        result_deepfake.append(label_str)
    else :
        # print(part, '0%,')
        result_deepfake.append(part + ' 0%,')


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
