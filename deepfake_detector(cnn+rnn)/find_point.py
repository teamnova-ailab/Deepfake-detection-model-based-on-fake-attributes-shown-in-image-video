
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from matplotlib import pyplot as plt
from collections import OrderedDict
import os




def cut_img(face_part_list, filter_image, facepart_index, shape, additional_point):

    # ---------------------------------------------------------------------------------------
    '''지정한 영역만 자르기(필터,원본 모두 필요)'''
    # 지정한 영역 입력 받기
    if face_part_list:
        # print(face_part_list)
        face_part = face_part_list.split(',')

        # 지정한 영역만큼 돌면서 자르기
        for (_, pts_name) in enumerate(face_part):
            index_name = pts_name
            pts = np.zeros((len(facepart_index[index_name]), 2), np.int32)
            for i, j in enumerate(facepart_index[index_name]):
                if j <= 68:
                    pts[i] = [shape[j][0], shape[j][1]]
                else:
                    pts[i] = [additional_point[j][0], additional_point[j][1]]

            # 마스크로 영역 지정
            filter_mask = np.zeros(filter_image.shape[:2], np.uint8)
            cv2.drawContours(filter_mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            # 검은색 배경으로 자르기
            black_background_filter = cv2.bitwise_and(filter_image, filter_image, mask=filter_mask)

    return black_background_filter


def cut_img_cheekline(face_part_list, filter_image, facepart_index, shape, additional_point):
    # ---------------------------------------------------------------------------------------
    '''지정한 영역만 자르기(필터,원본 모두 필요)'''
    # 지정한 영역 입력 받기
    if face_part_list:
        # print(face_part_list)
        face_part = face_part_list.split(',')
        # print(face_part)
        face_part_num = len(face_part)
        tmp_x_max1 = 0
        tmp_x_min1 = 0
        tmp_x_max2 = 0
        tmp_x_min2 = 0

        # 지정한 영역만큼 돌면서 자르기
        for (_, pts_name) in enumerate(face_part):
            index_name = pts_name
            pts = np.zeros((len(facepart_index[index_name]), 2), np.int32)
            for i, j in enumerate(facepart_index[index_name]):
                if j <= 68:
                    pts[i] = [shape[j][0], shape[j][1]]
                else:
                    pts[i] = [additional_point[j][0], additional_point[j][1]]

                # 왼쪽 볼
                if i == 0:
                    tmp_x_max1 = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0], pts[4][0], pts[5][0])
                    tmp_x_min1 = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0], pts[4][0], pts[5][0])
                # 오른쪽 볼
                else:
                    tmp_x_max2 = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0], pts[4][0], pts[5][0])
                    tmp_x_min2 = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0], pts[4][0], pts[5][0])


                # 마스크로 영역 지정
                filter_mask = np.zeros(filter_image.shape[:2], np.uint8)
                cv2.drawContours(filter_mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

                # 검은색 배경으로 자르기
                black_background_filter = cv2.bitwise_and(filter_image, filter_image, mask=filter_mask)

                # 흰색 배경으로 자르기
                bg_filter = np.ones_like(filter_image, np.uint8) * 255
                cv2.bitwise_not(bg_filter, bg_filter, mask=filter_mask)
                white_backgroud_filter = bg_filter + black_background_filter

                # 왼쪽 이미지는 left1,left2 오른쪽 이미지는 right1,right2에 저장해서 width를 비교한후 저장한다.
                if i == 0:
                    tmp_white_backgroud_filter1 = white_backgroud_filter.copy()
                elif i == 1:
                    tmp_white_backgroud_filter2 = white_backgroud_filter.copy()

            if abs(tmp_x_max1 - tmp_x_min1) > abs(tmp_x_max2 - tmp_x_min2):
               if abs(tmp_x_max1 - tmp_x_min1)  > abs(tmp_x_max2 - tmp_x_min2) and abs(tmp_x_max1 - tmp_x_min1)*0.8  < abs(tmp_x_max2 - tmp_x_min2):
                  # continue
                  pass
               else:
                  white_backgroud_filter = tmp_white_backgroud_filter1

            elif abs(tmp_x_max1 - tmp_x_min1) < abs(tmp_x_max2 - tmp_x_min2):
               if abs(tmp_x_max1 - tmp_x_min1)  < abs(tmp_x_max2 - tmp_x_min2) and abs(tmp_x_max1 - tmp_x_min1)  > abs(tmp_x_max2 - tmp_x_min2)*0.8:
                  # continue
                  pass
               else:
                  white_backgroud_filter = tmp_white_backgroud_filter2

    return white_backgroud_filter
