from detect_function import detect_face
from detection_filter3 import detect_face_filter3
from detect_deepfake import detect_face_deepfake
#from find_face_part import f_p_load_models, detect_face_part
#from find_fake import load_fake_models, detect_fake
import os
import time
import pandas
from utils import label_map_util
import glob
import tensorflow as tf
import cv2
import numpy as np
import threading

import dlib

import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import csv

'''
모델을 세션에 로드, 각 파트별 cnn 스레드 시작, rnn 결과 도출
'''

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def load_face_model(PATH_TO_CKPT):
    print(PATH_TO_CKPT)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return sess, detection_graph


code_start = time.time()
# 전체 모델 세션에 업로드
f_sess, f_detection_graph  = load_face_model('face_model/frozen_inference_graph_face.pb')
eye_sess, eye_detection_graph = load_face_model('face_part_model/eyes_v6.pb')
nose_sess, nose_detection_graph = load_face_model('face_part_model/nose_ssd_inception.pb')
mouth_sess, mouth_detection_graph = load_face_model('face_part_model/z_mouth_v6_ssd.pb')

f_d_sess, f_d_detection_graph = load_face_model('models/face_dotnoise_v11_ssd.pb')
f_g_sess, f_g_detection_graph = load_face_model('models/face_gridnoise_v7_ssd.pb')
f_blur_sess, f_blur_detection_graph = load_face_model('models/face_blur_faster.pb')

e_d_sess, e_d_detection_graph = load_face_model('models/eye_dotnoise_v1_ssd.pb')
e_g_sess, e_g_detection_graph = load_face_model('models/eye_gridnoise_v2_ssd.pb')
# e_glasses_sess, e_glasses_detection_graph = load_face_model('keep/eye_glasses_v2_ssd.pb')

n_d_sess, n_d_detection_graph = load_face_model('models/nose_dotnoise_v3_ssd_50k.pb')
n_g_sess, n_g_detection_graph = load_face_model('models/nose_gridnoise_v7_ssd.pb')
# n_bridge_sess, n_bridge_detection_graph = load_face_model('keep/nose_bridge_v3_ssd.pb')

m_d_sess, m_d_detection_graph = load_face_model('models/mouth_dotnoise_v3_ssd.pb')
m_g_sess, m_g_detection_graph = load_face_model('models/mouth_gridnoise_v2_ssd.pb')
# m_noteeth_sess, m_noteeth_detection_graph = load_face_model('keep/mouth_noteeth_v1_ssd.pb')


e_b_sess, e_b_detection_graph = load_face_model('models_3/eye_bridge_v4_ssd.pb')
e_t_l_sess, e_t_l_detection_graph = load_face_model('models_3/eye_temple_left_v2_ssd.pb')
e_t_r_sess, e_t_r_detection_graph = load_face_model('models_3/eye_temple_right_v2_ssd.pb')
n_c_sess, n_c_detection_graph = load_face_model('models_3/nose_cheekline_v2_ssd.pb')

alae_sess, alae_detection_graph = load_face_model('models_deepfake/alae_model_v3.pb')
stargan_sess, stargan_detection_graph = load_face_model('models_deepfake/stargan_model.pb')
stylegan_sess, stylegan_detection_graph = load_face_model('models_deepfake/stylegan_model.pb')
faceswap_sess, faceswap_detection_graph = load_face_model('models_deepfake/faceswap_model.pb')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# =============================================================================================
# rnn model 로드

fst_model = BertForSequenceClassification.from_pretrained('../rnn_data/data/model/e4/', num_labels=2)
trd_model = BertForSequenceClassification.from_pretrained('../rnn_data/data/model/y7/', num_labels=2)
deepfake_model = BertForSequenceClassification.from_pretrained('../rnn_data/data/model/deepfake_model/1/', num_labels=2)
img_model = BertForSequenceClassification.from_pretrained('../rnn_data/data/model/img_model/2/', num_labels=2)
video_model = BertForSequenceClassification.from_pretrained('../rnn_data/data/model/h1/', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)


# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    fst_model.cuda()
    trd_model.cuda()
    img_model.cuda()
    video_model.cuda()
else:
    device = torch.device("cpu")

fst_model.eval()
trd_model.eval()
img_model.eval()
video_model.eval()

''' 텍스트를 첫번째 bert 모델에 적합한 형태로 변환 및 라벨+퍼센트 출력'''
# 입력 데이터 변환 - 1차 필터
def fst_convert_input_data(fst_sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in fst_sentences]
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 80
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    # 데이터를 파이토치의 텐서로 변환
    fst_inputs = torch.tensor(input_ids)
    fst_masks = torch.tensor(attention_masks)
    return fst_inputs, fst_masks


# 라벨+퍼센트 출력 - 1차 필터
def fst_test_sentences(fst_sentences):
    inputs, masks = fst_convert_input_data(fst_sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = fst_model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu()

    label = logits.numpy()
    label = np.argmax(label)
    percent = F.softmax(logits, dim=1)
    percent = percent.data.numpy().squeeze()
    percent = np.expand_dims(percent, axis=0)

    if label == 0:
        percent = 1 - percent[0][0]
    else:
        percent = percent[0][1]

    return percent

''' 텍스트를 세번째 bert 모델에 적합한 형태로 변환 및 라벨+퍼센트 출력'''
# 입력 데이터 변환 - 3차 필터
def trd_convert_input_data(trd_sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in trd_sentences]
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 80
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    # 데이터를 파이토치의 텐서로 변환
    trd_inputs = torch.tensor(input_ids)
    trd_masks = torch.tensor(attention_masks)
    return trd_inputs, trd_masks


# 라벨+퍼센트 출력 - 3차 필터
def trd_test_sentences(trd_sentences):
    inputs, masks = trd_convert_input_data(trd_sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = trd_model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu()

    label = logits.numpy()
    label = np.argmax(label)
    percent = F.softmax(logits, dim=1)
    percent = percent.data.numpy().squeeze()
    percent = np.expand_dims(percent, axis=0)

    if label == 0:
        percent = 1 - percent[0][0]
    else:
        percent = percent[0][1]
    return percent

''' 텍스트를 딥페이크 bert 모델에 적합한 형태로 변환 및 라벨+퍼센트 출력'''
# 입력 데이터 변환 - deepfake 필터
def deepfake_convert_input_data(deepfake_sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in deepfake_sentences]
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 80
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    # 데이터를 파이토치의 텐서로 변환 
    deepfake_inputs = torch.tensor(input_ids)
    deepfake_masks = torch.tensor(attention_masks)
    return deepfake_inputs, deepfake_masks


# 라벨+퍼센트 테스트 - deepfake 필터
def deepfake_test_sentences(deepfake_sentences):
    deepfake_model.eval()
    inputs, masks = deepfake_convert_input_data(deepfake_sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = deepfake_model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu()
    label = logits.numpy()
    label = np.argmax(label)
    percent = F.softmax(logits, dim=1)
    percent = percent.data.numpy().squeeze()
    percent = np.expand_dims(percent, axis=0)

    if label == 0:
        percent = 1-percent[0][0]
    else:
        percent = percent[0][1]
    return percent


''' 텍스트를 이미지 판단 bert 모델에 적합한 형태로 변환 및 라벨+퍼센트 출력'''
# 입력 데이터 변환 - img 모델
def img_convert_input_data(final_sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in final_sentences]
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 100
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    # 데이터를 파이토치의 텐서로 변환
    final_inputs = torch.tensor(input_ids)
    final_masks = torch.tensor(attention_masks)
    return final_inputs, final_masks


# 라벨+퍼센트 출력 - img 모델
def img_test_sentences(final_sentences):
    inputs, masks = img_convert_input_data(final_sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = img_model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu()

    label = logits.numpy()
    label = np.argmax(label)
    percent = F.softmax(logits, dim=1)
    percent = percent.data.numpy().squeeze()
    percent = np.expand_dims(percent, axis=0)

    if label == 0:
        percent = 1 - percent[0][0]
    else:
        percent = percent[0][1]
    return percent

''' 텍스트를 비디오 판단 bert 모델에 적합한 형태로 변환 및 라벨+퍼센트 출력'''
# 입력 데이터 변환 - video 모델
def video_convert_input_data(final_sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in final_sentences]
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 100
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    # 데이터를 파이토치의 텐서로 변환
    final_inputs = torch.tensor(input_ids)
    final_masks = torch.tensor(attention_masks)
    return final_inputs, final_masks


# 라벨+퍼센트 출력 - img 모델
def video_test_sentences(final_sentences):
    inputs, masks = video_convert_input_data(final_sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = img_model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu()

    label = logits.numpy()
    label = np.argmax(label)
    percent = F.softmax(logits, dim=1)
    percent = percent.data.numpy().squeeze()
    percent = np.expand_dims(percent, axis=0)

    if label == 0:
        percent = 1 - percent[0][0]
    else:
        percent = percent[0][1]
    return percent



# =============================================================================================
''' rnn 모델 결과 도출 '''

#입력값 들어왔을 때
#1. 입력값 이미지, 동영상 구분
#2. 입력값이 이미지일때
#3. 입력값이 동영상일때

#csv파일 만들기
f = open('./txt_file/final_file/fake_4.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['filename', 'label'])


# 이미지, 동영상 확장자 별로 구분해서 배열에 담기
file_dir = '/home/teamnova/deepfake_detection/train_data/test_fake/4/**'
txt_output_dir = './txt_file/rnn2_fake/'

# 한 폴더만 가져오기 (dir/*)
#file_list = glob.glob(file_dir)
# 하위 폴더도 가져오기 (dir/**)
file_list = glob.glob(file_dir, recursive=True)

img_list = [file for file in file_list if file.endswith(".jpg") or file.endswith(".png") ]
video_list = [file for file in file_list if file.endswith(".mp4") or file.endswith(".avi") ]

n_img = len(img_list)
n_video = len(video_list)

print('n_img', n_img, 'n_video', n_video)

# 영상 라벨 - real 인지 fake 인지 입력
# fake = 0 / real  = 1
# final_label = 1
final_label = 0


# csv 저장할 배열 생성
label_list = []
label_list = "video_name/label+percent/video_tag\n"

start = time.time()


# ===========================================================================

'''
cnn detector 클래스 
-> cnn 탐지 + 가짜특징 판정 rnn(1차, 3차, 딥페이크 rnn) 
return : 가짜특징 판정 rnn 결과값
'''
def CNN_Detector(img, img_name) :

    # cnn detector 결과값 담을 배열 생성

    result_rnn = []
    result_face = []
    result_rnn_3 = []
    result_deepfake = []

    # 1번 섹션 스레드 : 눈코입에 나타나는 특징 (detect_function)
    th = threading.Thread(target=detect_face,
                          args=(result_rnn, result_face, img, f_sess, f_detection_graph,
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
                                m_g_sess, m_g_detection_graph
                                ))

    # 2번 섹션 스레드 : 얼굴 일부분에만 나타나는 특징 (detection_filter3)
    th_part = threading.Thread(target=detect_face_filter3,
                               args=(result_rnn_3, img,
                                     detector, predictor,
                                     f_sess, f_detection_graph,
                                     e_b_sess, e_b_detection_graph,
                                     e_t_l_sess, e_t_l_detection_graph,
                                     e_t_r_sess, e_t_r_detection_graph,
                                     n_c_sess, n_c_detection_graph))

    # 3번 섹션 스레드 : 페이스북 이외 데이터에서 나타나는 특징 (detection_deepfake)
    th_deepfake = threading.Thread(target=detect_face_deepfake,
                                   args=(result_deepfake, img, f_sess, f_detection_graph,
                                         alae_sess, alae_detection_graph,
                                         stylegan_sess, stylegan_detection_graph,
                                         stargan_sess, stargan_detection_graph,
                                         faceswap_sess, faceswap_detection_graph
                                         ))

    # 스레드로 cnn detector 동작
    th.start()
    th_part.start()
    # th_deepfake.start()
    th.join()
    th_part.join()
    # th_deepfake.join()

    # video rnn train data
    str_result = ""
    str_result_3 = ""
    str_result_deepfake = ""

    # 정면, 측면 파악해서 결과값에 추가
    #     print('result_face',result_face)
    if result_face == [''] or result_face == []:
        result_face.clear()
        result_face.append('unknown')
        # fake = 0 real = 1
        # print(vn,'/', img_num,'/', str_result,'/',str(final_label))
    # 1st filter detector result

    result_rnn.insert(0, str(result_face[0]))

    # 첫번째 cnn 탐지 모델 결과
    if result_rnn is not None:
        #    print('filter1 result', result_rnn)
        #    for i in range(len(result_rnn)):
        #         str_result += (result_rnn[i])
        # label_list1 +=  (str(img_name) + '/' + str(str_result) + '/' + str(final_label)) + "\n"

        len_result_rnn = len(result_rnn)
        if len(result_rnn) != 1:
            sentence_add = ''
            r_face_dot_noise = ''
            r_face_grid_noise = ''
            r_nose_dot_noise = ''
            r_nose_grid_noise = ''
            r_eye_dot_noise = ''
            r_face_notblur = ''
            r_mouth_dot_noise = ''

            for i in range(len_result_rnn):
                sentence_test = result_rnn[i]
                if sentence_test.find('face_dotnoise') != -1:
                    r_face_dot_noise = result_rnn[i][:-1] + ' '
                elif sentence_test.find('face_gridnoise') != -1:
                    r_face_grid_noise = result_rnn[i][:-1] + ' '
                elif sentence_test.find('nose_dotnoise') != -1:
                    r_nose_dot_noise = result_rnn[i][:-1] + ','
                elif sentence_test.find('nose_gridnoise') != -1:
                    r_nose_grid_noise = result_rnn[i][:-1] + ' '
                elif sentence_test.find('mouth_dotnoise') != -1:
                    r_mouth_dot_noise = result_rnn[i][:-1] + ','
                elif sentence_test.find('eye_dotnoise') != -1:
                    r_eye_dot_noise = result_rnn[i][:-1] + ' '
                if sentence_test.find('face_notblur') != -1:
                    r_face_notblur = result_rnn[i][:-1] + ' '

            sentence_add = result_rnn[
                               0] + ',' + r_face_notblur + r_face_grid_noise + r_nose_dot_noise + r_nose_grid_noise + r_mouth_dot_noise + r_eye_dot_noise + r_face_dot_noise
            # print(str(video_item), result_rnn)
            #                     print('not unknown',result_rnn, sentence_add)

            # 1차 rnn모델로 퍼센트 출력
            fst_logits = fst_test_sentences([sentence_add])
        # print(result_rnn,sentence_add, fst_logits)
        else:
            fst_logits = 'not_detect'
        # print('unknown',result_rnn,fst_logits)

    # 세번째 cnn 탐지 모델 결과
    if result_rnn_3 is not None:
        # print('filter3 result', result_rnn_3)

        # for i in range(len(result_rnn_3)):
        #      str_result_3 += (result_rnn_3[i])
        # label_list2 +=  (str(img_name) + '/' + str(str_result_3) + '/' + str(final_label)) + "\n"

        # 3차필터 라벨 순서 변경
        if len(result_rnn_3) != 0:
            len_result_rnn_v3 = len(result_rnn_3)
            r_eye_bridge = ''
            r_eye_temple_left = ''
            r_eye_temple_right = ''
            r_cheekline = ''
            r_mouth_black_line = ''

            for i in range(len_result_rnn_v3):
                sentence_test = result_rnn_3[i]
                if sentence_test.find('eye_bridge') != -1:
                    r_eye_bridge = result_rnn_3[i][:-1] + ' '
                elif sentence_test.find('eye_temple_left') != -1:
                    r_eye_temple_left = result_rnn_3[i][:-1] + ','
                elif sentence_test.find('eye_temple_right') != -1:
                    r_eye_temple_right = result_rnn_3[i][:-1] + ' '
                elif sentence_test.find('cheekline') != -1:
                    r_cheekline = result_rnn_3[i][:-1]
                elif sentence_test.find('mouth_black_line') != -1:
                    r_mouth_black_line = result_rnn_3[i][:-1] + ','

            sentence_add = r_eye_bridge + r_eye_temple_right + r_eye_temple_left + r_mouth_black_line + r_cheekline
            # print(str(video_item), result_rnn_v3)
            # print(str(video_item), sentence_add)

            # 3차 rnn모델로 퍼센트 출력
            trd_logits = trd_test_sentences([sentence_add])
        # print(result_rnn_v3,sentence_add, trd_logits)
        else:
            trd_logits = 'not_detect'
        # print(result_rnn_v3,trd_logits)
        # print(fst_logits,trd_logits)

    # deepfake cnn 탐지 모델 결과
    if result_deepfake is not None:
        # print('deepfake_result', result_deepfake)
        # for i in range(len(result_deepfake)):
        #     str_result_deepfake += (result_deepfake[i])
        # print('deepfake_result', (str(img_name) + '/' + str(str_result_deepfake) + '/' + str(final_label)))
        if len(result_deepfake) != 0:
            len_result_deepfake = len(result_deepfake)
            alae = ''
            stylegan = ''
            stargan = ''
            faceswap = ''

            for i in range(result_deepfake):
                sentence_test = result_deepfake[i]
                if sentence_test.find('alae') != -1:
                    alae = result_deepfake[i] + ' '
                if sentence_test.find('stylegan') != -1:
                    stylegan = result_deepfake[i] + ','
                if sentence_test.find('stargan') != -1:
                    stargan = result_deepfake[i] + ' '
                if sentence_test.find('faceswap') != -1:
                    faceswap = result_deepfake[i] + ','
                if sentence_test.find('began') != -1:
                    faceswap = result_deepfake[i] + ','
                if sentence_test.find('mwgan') != -1:
                    faceswap = result_deepfake[i] + ','
                if sentence_test.find('casualgan') != -1:
                    faceswap = result_deepfake[i] + ','
                if sentence_test.find('hat') != -1:
                    faceswap = result_deepfake[i] + ','

            sentence_add = stylegan + stargan + faceswap + alae

            # deepfake rnn모델로 퍼센트 출력
            deepfake_logits = deepfake_test_sentences([sentence_add])
        # print(result_deepfake,sentence_add, deepfake_logits)
        else:
            deepfake_logits = 'not_detect'

    return fst_logits, trd_logits, deepfake_logits

# ===========================================================================

''' 이미지 탐지 '''


for img_file in img_list :
    label_add = ''
    # 이미지 읽어오기
    img = cv2.imread(img_file)
    img_name = img_file.split('/')[-1]

    fst_logits, trd_logits, deepfake_logits = CNN_Detector(img_file, img_name)


         # 데이터 합치기
    label_add = '1st filter ' + str(fst_logits) + ' 3rd filter ' + str(trd_logits)  + ', deepfake filter ' + str(deepfake_logits)

         # 이미지 rnn 모델로 퍼센트 출력
    final_logits = img_test_sentences([label_add])


      # 결과값 확인
    print(str(img_name) + "/" + str(final_logits))
  
      # 결과값 csv파일에 담기
    wr.writerow([str(img_name),final_logits])

   # cv2.imshow('Object detector', img)
   # cv2.waitKey(0)
                # Press 'q' to quit
                #if cv2.waitKey(1) == ord('q'):
                #    break


# txt = open(txt_output_dir + 'faceswap_1' + '.txt', 'w', encoding='utf-8', newline='')
# txt.write(label_list1)
# txt.close()

# txt = open(txt_output_dir + 'faceswap_2' + '.txt', 'w', encoding='utf-8', newline='')
# txt.write(label_list2)
# txt.close()

# ===========================================================================


''' 비디오 탐지 '''


for PATH_TO_VIDEO in video_list :

  #  folder_list = os.listdir(folder_path)
        print('folder_path', PATH_TO_VIDEO)


    # ===========================================================================
    # num = input("영상에서 몇 프레임마다 detect 할지 입력 : ")
    # count = input("몇번째 프레임을 확인할지 입력 : ")
        num = 30
        count = 15
        start = time.time()

        get_video_name = PATH_TO_VIDEO.split('.')
        if len(get_video_name) == 3 :
           video_name = PATH_TO_VIDEO.split('.')[1]
        else :
           video_name = PATH_TO_VIDEO.split('.')[0]	
        vn = video_name.split('/')[-1]
	
        print('video_name',video_name,'vn',vn)

        cap = cv2.VideoCapture(PATH_TO_VIDEO)

        #동영상 저장 준비
        #print('fps', cv2.CAP_PROP_FPS, 'frame_num', cv2.CAP_PROP_FRAME_COUNT)
        #print('size', cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #writer = cv2.VideoWriter('test2.avi',fourcc,10.0, (width, height))

        frame_num = 300
        label_add = ''

        while frame_num:
            
            result_rnn = []
            result_face =[]
            result_rnn_3 =[]
            result_deepfake = []

            ret, frame = cap.read()
            # face_result = list()
            if ret == 0:
                break
            if (int(cap.get(1)) % num == count):
                img_num = int(cap.get(1))

                fst_logits, trd_logits, deepfake_logits = CNN_Detector(img_file, img_name)
                img_label = '1st filter ' + str(fst_logits) + ' 3rd filter ' + str(trd_logits)
                label_add = label_add + img_test_sentences([img_label])

                #동영상 저장
                #writer.write(frame)

                #cv2.imshow('Object detector', frame)
                # Press 'q' to quit
                #if cv2.waitKey(1) == ord('q'):
                #    break
            frame_num -= 1

        final_logits = img_test_sentences([label_add])

        # 결과값 확인
        print(str(img_name) + "/" + str(final_logits))

        txt = open(txt_output_dir + vn + '.txt', 'w', encoding='utf-8', newline='')
        txt.write(label_list)
        txt.close()


print('model process time ', time.time() - start)
print('process time ', time.time() - code_start)
