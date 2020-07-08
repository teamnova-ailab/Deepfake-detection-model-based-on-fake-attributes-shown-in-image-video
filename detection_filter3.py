import cv2
import threading
import numpy as np
import filter_def
from imutils import face_utils
import numpy as np
import cv2
from collections import OrderedDict

from utils import visualization_utils_face_v3
from utils import visualization_utils_fake

from find_point import cut_img, cut_img_cheekline
    # , cut_img_cheekline

'''
스레드로 각 특징별 탐지 모델 동작시키고 결과값을 배열에 담아서 반환 (탐지 모델 동작에서 시각화 및 시각화에 필요한 부분은 visualization 에 있음)
 - 얼굴 일부분에만 나타나는 특징
'''

# 얼굴 찾는 클래스 -> 얼굴 탐지 후 가짜특징에 맞게 얼굴 영역 분할
def detect_face_filter3(result_rnn_3, frame,
                        detector, predictor,
                        sess, detection_graph,
                        e_b_sess, e_b_detection_graph,
                        e_t_l_sess, e_t_l_detection_graph,
                        e_t_r_sess, e_t_r_detection_graph,
                        n_c_sess, n_c_detection_graph):

    # 얼굴 찾기
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

    left, right, top, bottom, label_str = \
        visualization_utils_face_v3.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        use_normalized_coordinates=True,
        line_thickness=4)

    # print(video_item, label_str)
    image = frame[int(top):int(bottom), int(left):int(right)]
    height = bottom - top
    width = right - left
    if left != 0 and height > 0 and width > 0:
        # print('find_face')

        # 얼굴 포인트 찾기
        th_find_point = threading.Thread(target=find_face_point, args = (result_rnn_3, image, detector, predictor,
                                                         e_b_sess, e_b_detection_graph, e_t_l_sess, e_t_l_detection_graph,
                                                         e_t_r_sess, e_t_r_detection_graph, n_c_sess, n_c_detection_graph))
        th_find_point.start()
        th_find_point.join()

    # 얼굴 이미지당 한번만 실행

# 각 특징별 부위를 찾는 클래스 (68랜드마크 활용)
def find_face_point(result_rnn_3, image, detector, predictor,
                    e_b_sess, e_b_detection_graph,
                    e_t_l_sess, e_t_l_detection_graph, e_t_r_sess, e_t_r_detection_graph,
                    n_c_sess, n_c_detection_graph):

    h, w, c = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)

        # 선택된 영역이 이미지 넓이를 벗어난 경우 예외처리
        left = rect.left()
        top = rect.top()
        if left < 0:
            left = 0
        elif top < 0:
            top = 0
        shape = face_utils.shape_to_np(shape)

        # 나누는 값이 0인지를 check하는 코드
        tmp_check = (shape[17][0] - shape[19][0])
        if tmp_check == 0:
            tmp_check = 1

        # ---------------------------------------------------------------------------------------
        ''' 추가 포인트 찍기 (포인트 확정된 후 필요없는 포인트는 지울것)'''

        # 눈썹 6개 왼쪽 - 69, 70, 71 / 오른쪽 - 72,73,74
        # 광대 2개 - 75, 76
        # 광개 옆 2개 - 77, 78
        # 입 주변 5개 - 79, 80, 81, 82, 83
        # 미간 1개 - 84
        # 광대위 2개 - 85,86
        # 눈썹옆 2개 - 87,88
        # 턱선 밑 9개 - 89,90,91,92,93,94,95,96,97
        # 귀 4개 - 98,99,100,101
        # 입꼬리 옆 2개 - 102,103
        # 눈썹 위 9개 - 104,105,106,107,108,109,110,111,112

        # shape 좌표 접근 shape[랜드마크번호][x:0,y:1]

        # 눈썹 6개 왼쪽 - 69, 70, 71 / 오른쪽 - 72,73,74
        # 69번 : 17, 19번의 선에서 18이 만나는 곳의 길이만큼 17번에서 올리기
        # y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        part1_m = (shape[17][1] - shape[19][1]) / tmp_check
        part1_n = shape[17][1] - (part1_m * shape[17][0])
        part1_y = part1_m * shape[18][0] + part1_n
        part1_d = abs(shape[18][1] - part1_y)

        x_70 = shape[19][0]
        y_70 = int(shape[19][1] - part1_d)
        x_71 = shape[21][0]
        y_71 = int(shape[21][1] - part1_d)

        # 나누는 값이 0인지를 check하는 코드
        tmp_check = (shape[17][0] - shape[19][0])
        if tmp_check == 0:
            tmp_check = 1

        part5_m = (shape[26][1] - shape[24][1]) / tmp_check
        # print(part5_m)
        part5_n = shape[26][1] - (part5_m * shape[26][0])
        part5_y = part5_m * shape[25][0] + part5_n
        part5_d = abs(shape[25][1] - part5_y)

        x_72 = shape[22][0]
        y_72 = int(shape[22][1] - part5_d)
        x_73 = shape[24][0]
        y_73 = int(shape[24][1] - part5_d)

        # 광대 2개 - 75, 76
        # 75 : 3과 29의 중간 , 76 : 13과 29의 중간
        x_75 = int(round((shape[3][0] + shape[29][0]) / 2))
        y_75 = int(round((shape[3][1] + shape[29][1]) / 2))
        x_76 = int(round((shape[13][0] + shape[29][0]) / 2))
        y_76 = int(round((shape[13][1] + shape[29][1]) / 2))

        # 광개 옆 2개 - 77, 78
        # 77 : 2와 41 중간, 78 : 46과 14 중간
        x_77 = int(round((shape[2][0] + shape[41][0]) / 2))
        y_77 = int(round((shape[2][1] + shape[41][1]) / 2))
        x_78 = int(round((shape[14][0] + shape[46][0]) / 2))
        y_78 = int(round((shape[14][1] + shape[46][1]) / 2))

        # 광대위 2개 - 85,86
        # 85 : 39, 75 / 86 : 42, 76
        x_85 = int(round((shape[39][0] + x_75) / 2))
        y_85 = int(round((shape[39][1] + y_75) / 2))
        x_86 = int(round((shape[42][0] + x_76) / 2))
        y_86 = int(round((shape[42][1] + y_76) / 2))

        # 턱선 밑 9개 - 89,90,91,92,93,94,95,96,97
        # x2,y2가 x1,y1와 x3,y3의 중간일 때 x1 = 2*x2-x3 y1 = 2*y2-y3
        x_89 = int(2 * shape[3][0] - (x_75 + shape[3][0]) / 2)
        y_89 = int(2 * shape[3][1] - (y_75 + shape[3][1]) / 2)

        # 귀 4개 - 98,99,100,101
        x_98 = 0
        y_98 = y_70
        x_99 = 0
        y_99 = shape[2][1]
        x_100 = w
        y_100 = y_73
        x_101 = w
        y_101 = shape[14][1]

        # ---------------------------------------------------------------------------------------
        ''' 추가 점 배열에 담기'''
        additional_point = OrderedDict(
            [(70, (x_70, y_70)), (71, (x_71, y_71)), (72, (x_72, y_72)), (73, (x_73, y_73)), (75, (x_75, y_75)),
             (76, (x_76, y_76)), (77, (x_77, y_77)), (78, (x_78, y_78)), (85, (x_85, y_85)), (86, (x_86, y_86)),
             (89, (x_89, y_89)), (98, (x_98, y_98)), (99, (x_99, y_99)), (100, (x_100, y_100)), (101, (x_101, y_101))])

        # ---------------------------------------------------------------------------------------
        '''영역별로 나눠서 배열에 담기'''
        facepart_index = OrderedDict(
            [("cheek_line1", (19, 38, 75, 60, 89, 98)),
             ("cheek_line2", (24, 45, 76, 54, 101, 100)),
             ("eye_bridge", (20, 38, 40, 85, 29, 86, 47, 43, 23, 72, 71)),
             ("eye_temple_left", (98, 70, 37, 41, 77, 2, 99)),
             ("eye_temple_right", (100, 73, 44, 46, 78, 14, 101))])

        # 가짜 특징 찾는 스레드
        th_eye_bridge = threading.Thread(target=detect_fake,
                                         args=(result_rnn_3, 'eye_bridge', facepart_index, shape, additional_point,
                                               image, e_b_sess, e_b_detection_graph))

        th_eye_temple_left = threading.Thread(target=detect_fake,
                                              args=(result_rnn_3, 'eye_temple_left', facepart_index, shape, additional_point,
                                                  image, e_t_l_sess, e_t_l_detection_graph))

        th_eye_temple_right = threading.Thread(target=detect_fake,
                                               args=(result_rnn_3, 'eye_temple_right', facepart_index, shape, additional_point,
                                                   image, e_t_r_sess, e_t_r_detection_graph))

        th_cheek_line = threading.Thread(target=detect_fake,
                                         args=(result_rnn_3, 'cheek_line1,cheek_line2', facepart_index, shape, additional_point,
                                             image, n_c_sess, n_c_detection_graph))

        th_eye_bridge.start()
        th_eye_bridge.join()

        th_eye_temple_left.start()
        th_eye_temple_left.join()

        th_eye_temple_right.start()
        th_eye_temple_right.join()

        th_cheek_line.start()
        th_cheek_line.join()


#각 가짜 특징에 맞는 필터를 적용, 각 특징별 탐지 모델 동작시키는 클래스
def detect_fake(result_rnn_3, part, facepart_index, shape, additional_point,
                    image, sess, detection_graph):
    # print(part)
    if 'cheek_line' in part:
        f_image = filter_def.cheekline(image)
        crop_image = cut_img_cheekline(part, f_image, facepart_index, shape, additional_point)

        part_list = []
        label_str = fake_model_run(crop_image, image, part, part_list, sess, detection_graph)

        if label_str != "":
            # print(label_str)
            result_rnn_3.append(label_str)
        else:
            result_rnn_3.append('cheekline' + ' 0%,')
            # print(part, '0%,')
    else:
        if 'bridge' in part :
            f_image = filter_def.eye_bridge(image)
        elif 'temple' in part :
            f_image = filter_def.eye_temple(image)

        crop_image = cut_img(part, f_image, facepart_index, shape, additional_point)

        part_list = []
        label_str = fake_model_run(crop_image, image, part, part_list, sess, detection_graph)
        if label_str != "":
            # print(label_str)
            result_rnn_3.append(label_str)
        else :
            result_rnn_3.append(part + ' 0%,')
            # print(part, '0%,')


# 가짜특징 탐지하는 클래스
def fake_model_run (f_image, image, part, part_list, sess, detection_graph) :
    # 마스크, 필터까지 다 거친 이미지 배열
    image_np_expanded = np.expand_dims(f_image, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
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
