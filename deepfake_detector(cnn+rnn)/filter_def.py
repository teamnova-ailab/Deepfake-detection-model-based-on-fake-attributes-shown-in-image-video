import cv2
import numpy as np

# 눈 필터
def eye(image) :
    tmp = image.copy()

    # Clahe (33)
    # RGB인 이미지를 Lab으로 바꾼다.
    # Lab = L(검점/흰색) A(초록/빨강) B(파랑/노랑)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2Lab)

    # 이미지에서 rgb값을 추출한다.
    l, a, b = cv2.split(tmp)

    # 세부변수인 ksize와 alpha값을 트랙바에서 가져온다.
    ksize = 1
    alpha = 1

    # CLAHE필터를 세부정보를 넣고 생성한다.
    clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(ksize, ksize))

    # ClAHE필터를 b에만 적용한다.
    dst = clahe.apply(l)

    # b를 CLAHE필터를 적용한 값으로 바꿔준다.
    l = dst.copy()

    # RGB를 합쳐서 3채널 이미지로 바꿔준다.
    tmp2 = cv2.merge((l, a, b))

    # Lab으로 변경한 이미지를 다시 RGB로 바꿔준다.
    work = cv2.cvtColor(tmp2, cv2.COLOR_LAB2BGR)

    # Emboss (32)
    work = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    work = cv2.filter2D(work, cv2.CV_8U, Kernel_emboss)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization (31)
    work = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(work, work)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # Bilateral
    work = cv2.bilateralFilter(work, 5, 255, 255)
    work = cv2.bilateralFilter(work, 5, 255, 255)
    work = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    return work


# 코 필터
def nose(image) :
    tmp = image.copy()

    # Gamma_correction
    gamma = 20
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    #CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(10, 10))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)

    #THRESH_TRUNC
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 180, 255, cv2.THRESH_TRUNC)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    image = tmp.copy()
    return image

# 입 필터
def mouth(image) :
    tmp = image.copy()

    # Gamma_correction
    gamma = 7
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(40, 40))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)

    # THRESH_TRUNC
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 255, 255, cv2.THRESH_TRUNC)
    # 3채널이미지로 바꿔준다.
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()

    return image

# 얼굴 가로 세로 노이즈
def face_gridnoise(image) :
    tmp = image.copy()
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # # #emboss필터 적용
    tmp = gray.copy()
    # work = tmp.copy()
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    work = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = work.copy()
    return image

# 얼굴 점 노이즈
def face_dotnoise(image) :
    tmp = image.copy()
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # Emboss
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel)

    # Median
    tmp = cv2.medianBlur(tmp, ksize=5)

    # Laplacian_of_Gaussian
    tmp = cv2.GaussianBlur(tmp, ksize=(5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    tmp = cv2.Laplacian(tmp, cv2.CV_16S, ksize=1, scale=4, delta=0,
                        borderType=cv2.BORDER_DEFAULT)
    LaplacianImage = cv2.convertScaleAbs(tmp)
    tmp = cv2.cvtColor(LaplacianImage, cv2.COLOR_GRAY2RGB)

    # THRESH_BINARY
    ret, tmp = cv2.threshold(tmp, 167, 255, cv2.THRESH_BINARY)
    # work = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    image = tmp.copy()
    return image

def face_blur(image) :

    # Brightness_measure
    tmp = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(tmp)
    b_mean, b_stddev = cv2.meanStdDev(v)
    mean_var = b_mean[0][0]
    tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB)
    #    cv2.putText(tmp, str(round(mean_var, 2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # 수치 60 이하일 경우, 60 이상일 경우 나눠서 필터 적용
    if mean_var > 60:
        # Sharpen
        alpha_30 = 9
        Kernel_sharpen = np.array([[0, -alpha_30, 0], [-alpha_30, 1 + 4 * alpha_30, -alpha_30], [0, -alpha_30, 0]])
        tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_sharpen)

        # Brightness&Contrast
        var_Brightness = 200 - 100
        var_Contrast = 15 - 100

        if (var_Contrast > 0):
            delta = 127.0 * var_Contrast / 100
            a = 255.0 / (255.0 - delta * 2)
            b = a * (var_Brightness - delta)
        else:
            delta = -128.0 * var_Contrast / 100
            a = (256.0 - delta * 2) / 255.0
            b = a * var_Brightness + delta

        tmp = tmp * a + b
        tmp = np.uint8(tmp)

        # Median
        tmp = cv2.medianBlur(tmp, ksize=3)

        # THRESH_BINARY
        ret, tmp = cv2.threshold(tmp, 123, 255, cv2.THRESH_BINARY)
        image = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

        image = image.copy()
    return image

# 안경 필터
def eye_glasses(image) :
    tmp = image.copy()

    # Gamma_correction (43)
    gamma = 20
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # High_pass (28)
    ksize_26 = 6
    kernel = np.array([[-0.5, -1, -0.5], [-1, ksize_26, -1], [-0.5, -1, -0.5]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, kernel)

    # Glasses From Paper
    # Laple_Kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    Gaussian_Kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    # tmp = cv2.filter2D(tmp, cv2.CV_16S, Laple_Kernel)
    tmp = cv2.filter2D(tmp, cv2.CV_16S, Gaussian_Kernel)
    # tmp = cv2.filter2D(tmp, cv2.CV_16S, Laple_Kernel)
    tmp = np.clip(tmp, 0, 255)
    tmp = np.uint8(tmp)

    work = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    return image

# 눈 격자 노이즈
def eye_gridnoise(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    work = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    return work


# 눈 점 노이즈
def eye_dotnoise(image):
    tmp = image.copy()
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    work = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    return work

# 코 노이즈
def nose_noise(image) :
    tmp = image.copy()

    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # # #emboss필터 적용
    tmp = gray.copy()
    # work = tmp.copy()
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    work = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = work.copy()

    return image

# 입 노이즈
def mouth_noise(image) :
    tmp = image.copy()
    # # # 그레이 처리를 한다
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # # #emboss필터 적용
    tmp = gray.copy()
    # work = tmp.copy()
    # Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    Kernel = np.array([[-21, -1, 0], [-1, 1, 1], [0, 1, 21]])
    work = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    image = work.copy()
    return image


# =========================================================================================================================
# 3차 필터 코드


# eyeline
def eye_line(image) :
    tmp = image.copy()

    # Histogram_Equalization (31)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    work = tmp.copy()
    return work


# eyebridge
def eye_bridge(image) :
    tmp = image.copy()

    # Histogram_Equalization (31)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Emboss (32)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Bilateral (22)
    tmp = cv2.bilateralFilter(tmp, 5, 153, 153)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # v3_bridge ksize_26 = 7
    # High_pass (28)
    ksize_26 = 7
    kernel = np.array([[-0.5, -1, -0.5], [-1, ksize_26, -1], [-0.5, -1, -0.5]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, kernel)

    # Gamma_correction (43)

    gamma = 90
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    work = tmp.copy()

    return work

# eye_temple
def eye_temple(image) :
    tmp = image.copy()

    # Histogram_Equalization (31)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    work = tmp.copy()

    return work

# cheekline
def cheekline(image) :

    tmp = image.copy()
    # CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(10, 10))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)

    # Histogram_Equalization (31)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Gamma_correction
    gamma = 100
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # Bilateral (22)
    tmp = cv2.bilateralFilter(tmp, 5, 153, 153)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # 이미지를 grayscale한다.
    # tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # 세부변수 High_Threshold, ksize, scale의 값을 트랙바에서 가져온다.
    high = 255
    ksize = 9
    scale = 10

    tmp = cv2.adaptiveThreshold(tmp, high, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, scale)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    work = tmp.copy()

    return work


# mouth_blackline
def mouth_blackline(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Gamma_correction
    gamma = 120
    # gamma = 60
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # Bilateral
    tmp = cv2.bilateralFilter(tmp, 5, 160, 160)

    # Prewitt
    Kernel_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    Kernel_Y = np.array([[0, 0, -1], [0, -1, 0], [0, 0, 0]])

    grad_x = cv2.filter2D(tmp, cv2.CV_16S, Kernel_X)
    grad_y = cv2.filter2D(tmp, cv2.CV_16S, Kernel_Y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    tmp = cv2.addWeighted(abs_grad_x, 4, abs_grad_y, 0, 0)

    tmp.astype(np.uint8)
    # tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    work = tmp.copy()

    return work

# alae 필터
def alae(image):
    tmp = image.copy()

    # HSV
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)


    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel = np.array([[-4, -1, 0], [-1, 1, 1], [0, 1, 4]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    work = tmp.copy()

    return work

# stylegan 필터
def stylegan(image):
    tmp = image.copy()

    alpha_30 = 4
    Kernel_sharpen = np.array([[0, -alpha_30, 0], [-alpha_30, 1 + 4 * alpha_30, -alpha_30], [0, -alpha_30, 0]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_sharpen)
    
    

    work = tmp.copy()

    return work

# stargan 필터
def stargan(image):
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-4, -1, 0], [-1, 1, 1], [0, 1, 4]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    work = tmp.copy()
    return work

# faceswap 필터
def faceswap(image):
    tmp = image.copy()

    # Sharpen
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    alpha_30 = 27
    Kernel_sharpen = np.array([[0, -alpha_30, 0], [-alpha_30, 1 + 4 * alpha_30, -alpha_30], [0, -alpha_30, 0]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_sharpen)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    work = tmp.copy()
    return work
