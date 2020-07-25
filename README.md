# [paper] Deepfake detection model based on fake attributes shown in image/video

## 20.07.25 수정중

### 딥페이크 탐지 동작 동영상
[![Video Label](https://img.youtube.com/vi/aixd8Egw-dA/hqdefault.jpg)](https://youtu.be/aixd8Egw-dA)

# Abstract
딥페이크는 딥러닝을 사용해서 만들어진 가짜 이미지를 의미한다. 딥페이크 비디오가 SNS를 비롯한 인터넷에 확산되면서 이와 관련된 사회문제가 발생하고 있다. [4,5,6] 딥페이크를 탐지할 수 있는 모델의 중요성이 대두되면서 많은 탐지모델이 제안되었지만, 딥페이크 생성 알고리즘 중 일부만 탐지할 수 있어 현실에 적용하기 어렵다. (표2 참조) 본 논문에서는 현실에 있는 다양한 딥페이크 생성 알고리즘에 대응할 수 있도록 CNN(Convolutional Neural Network)으로 가짜 특징을 탐지하고 RNN(Recurrent Neural Network)으로 가짜인지 판단하는 모델을 제안하며, 실험을 통해 제안된 구조의 탐지모델이 현실에서 사용될 수 있음을 입증하고자 한다. 갈수록 다양해지는 생성 알고리즘을 탐지 가능한지 입증하기 위해 탐지모델을 보완하면 탐지율이 향상되는지 실험을 진행했다. 또한, 다양한 딥페이크 생성 알고리즘에 대응하기 위해 여러 연구자의 참여가 필요하다. 이를 위해 제시된 구조로 딥페이크 탐지모델을 학습하여 탐지했을 때의 탐지율이 고사양 GPU를 사용한 모델[37, 38]과 비슷함을 증명하고자 했다. 실험 결과는 탐지모델 보완 후 Recall 2%에서 75%으로 증가, FPR 1.6%에서 0.1%으로 감소, AUC 0.02에서 0.77으로 증가하였으며, 고사양의 GPU를 사용한 모델[37, 38]과 비슷한 탐지율을 보였다. 

# 실험 결과
![결과그림_통합4](https://user-images.githubusercontent.com/44520048/88448935-d94a9900-ce7d-11ea-9f42-93ed932432c9.png)

# 코드 실행 방법

전체 탐지 모델(deepfake detector 폴더) : 
1. 폴더 생성 후  pb 파일추가 
2. 폴더 생성 후 pbtxt 파일추가
3. 실행할 .py 파일 안에 생성한 폴더 이름으로 모델 추가
3. 실행할 .py 파일 안에 이미지 가져올 폴더 경로 작성
4. 코드 실행 

<pre><code>python Main_thread.py</code></pre>

CNN Detector(cnn detector 폴더) : 

1. model 폴더 생성 후  pb 파일추가  
2. labelmap 폴더 생성 후 pbtxt 파일추가
3. 실행할 .py 파일 안에 이미지 가져올 폴더 및 탐지 결과 이미지 저장할 폴더 경로 작성
4. 코드 실행 

이미지 탐지 
<pre><code>python Object_detection_image.py</code></pre>
이미지 탐지  + csv 파일 생성 
<pre><code>python Object_detection_image.py > csv_file_name.csv</code></pre>

동영상 탐지
<pre><code>python Object_detection_video.py</code></pre>
동영상 탐지 + csv 파일 생성 
<pre><code>python Object_detection_video.py > csv_file_name.csv</code></pre>

# 학습 및 탐지 모델에 필요한 환경 및 라이브러리

tensorflow 1.15.0

tensorflow-gpu 1.15.0

transformers [[github]](https://github.com/huggingface/transformers)

tensorflow object detection api [[github]](https://github.com/tensorflow/models/tree/master/research/object_detection)


# CNN Detector 학습 방법 
 1. tensorflow object detection api를 설치합니다.
 2. 

# CNN Detector 다운로드 
: 딥페이크 탐지에 사용 가능한 CNN Detector를 다운받을 수 있습니다.

[Google drive](https://drive.google.com/drive/folders/1r7e7sbfoyVV-INuTLlTw8CoZvBESRWS5?usp=sharing)


# CNN Detector 발송 메일 주소 
: 연구한 CNN Detector로 딥페이크 탐지율을 향상시키는데 기여할 수 있습니다.

이메일 주소 : [teamnova.ailab@gmail.com](teamnova.ailab@gmail.com)

발송시 탐지한 deepfake 알고리즘 기재 및 .pb , .pbtxt , CNN Detector 테스트 파일 전송 부탁드립니다. 


# Contributor
: 딥페이크 탐지모델 탐지율 향상에 기여한 연구자 목록 

