import cv2
import numpy as np
import my_gaussian as gaussian

#low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1, pad_type='zero'):
    #########################################################################################
    # TODO                                                                                   #
    # apply_lowNhigh_pass_filter 완성                                                        #
    # Ix : image에 DoG_x filter 적용 or gaussian filter 적용된 이미지에 sobel_x filter 적용    #
    # Iy : image에 DoG_y filter 적용 or gaussian filter 적용된 이미지에 sobel_x filter 적용    #
    ###########################################################################################
    #low-pass filter를 이용하여 blur효과
    #high-pass filter를 이용하여 edge 검출
    #gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    DoG_x, DoG_y = get_my_DoG(fsize)
    Ix = gaussian.my_filtering(src, DoG_x, 'repetition')
    Iy = gaussian.my_filtering(src, DoG_y, 'repetition')

    return Ix, Iy

def get_my_DoG(msize, sigma=1):
    y, x = np.mgrid[-(msize//2):(msize//2)+1, -(msize//2):(msize//2)+1]

    DoG_x = (-x / sigma**2) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    DoG_y = (-y / sigma**2) * np.exp(-(x**2 + y**2)/(2*sigma**2))

    return DoG_x, DoG_y

#Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ##################################################
    # TODO                                           #
    # calcMagnitude 완성                             #
    # magnitude : ix와 iy의 magnitude를 계산         #
    #################################################
    dst_temp = np.sqrt(Ix ** 2 + Iy ** 2)
    # dst = (dst_temp).astype(np.uint8)
    # magnitude = dst/np.max(dst)
    # return magnitude
    return dst_temp

#Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    #######################################
    # TODO                                #
    # calcAngle 완성                      #
    # angle     : ix와 iy의 angle         #
    #######################################

    (h, w) = Ix.shape
    angle = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            angle[row, col] = np.arctan(Iy[row, col] / Ix[row, col])
    return angle

#non-maximum supression 수행ㅍ
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                      #
    # larger_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)         #
    ####################################################################################
    (h, w) = magnitude.shape
    larger_magnitude = np.zeros((h, w))
    angle_deg = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if (angle[i, j] < 0):
                angle_deg[i, j] = np.rad2deg(angle[i, j]) + 180
            else:
                angle_deg[i, j] = np.rad2deg(angle[i, j])

    for row in range(1, h-1):
        for col in range(1, w-1):
            try:
                m1 = 0
                m2 = 0
                # angle 0
                if(0 <= angle_deg[row, col] < 22.5) or (157.5 <= angle_deg[row, col] <= 180):
                    s = abs(np.tan(angle[row, col]))
                    if(angle_deg[row, col] == 0):
                        m1 = magnitude[row, col+1]
                        m2 = magnitude[row, col-1]
                    elif (0 < angle_deg[row, col] < 22.5):
                        m1 = magnitude[row, col + 1] * (1 - s) + magnitude[row + 1, col + 1] * s
                        m2 = magnitude[row, col - 1] * (1 - s) + magnitude[row - 1, col - 1] * s
                    elif (157.5 < angle_deg[row, col] <= 180):
                        m1 = magnitude[row, col + 1] * (1 - s) + magnitude[row - 1, col + 1] * s
                        m2 = magnitude[row, col - 1] * (1 - s) + magnitude[row + 1, col - 1] * s
                # angle 45
                elif (22.5 <= angle_deg[row, col] < 67.5):
                    if (angle_deg[row, col] == 45):
                        m1 = magnitude[row + 1, col + 1]
                        m2 = magnitude[row - 1, col - 1]
                    elif (22.5 < angle_deg[row, col] < 45):
                        s = abs(np.tan(angle[row, col]))
                        m1 = magnitude[row, col + 1] * (1 - s) + magnitude[row + 1, col + 1] * s
                        m2 = magnitude[row, col - 1] * (1 - s) + magnitude[row - 1, col - 1] * s
                    elif (45 < angle_deg[row, col] < 67.5):
                        s = 1/abs(np.tan(angle[row, col]))
                        m1 = magnitude[row + 1, col + 1] * s + magnitude[row + 1, col] * (1 - s)
                        m2 = magnitude[row - 1, col - 1] * s + magnitude[row - 1, col] * (1 - s)
                # angle 90
                elif (67.5 <= angle_deg[row, col] < 112.5):
                    s = 1/abs(np.tan(angle[row, col]))
                    if (angle_deg[row, col] == 90):
                        m1 = magnitude[row + 1, col]
                        m2 = magnitude[row - 1, col]
                    elif (67.5 < angle_deg[row, col] < 90):
                        m1 = magnitude[row + 1, col + 1] * s + magnitude[row + 1, col]*(1 - s)
                        m2 = magnitude[row - 1, col - 1] * s + magnitude[row - 1, col]*(1 - s)
                    elif (90 < angle_deg[row, col] < 112.5):
                        m1 = magnitude[row + 1, col - 1] * s + magnitude[row + 1, col]*(1 - s)
                        m2 = magnitude[row - 1, col + 1] * s + magnitude[row - 1, col]*(1 - s)
                # angle 135
                elif (112.5 <= angle_deg[row, col] < 157.5):
                    if (angle_deg[row, col] == 135):
                        m1 = magnitude[row + 1, col - 1]
                        m2 = magnitude[row - 1, col + 1]
                    elif (112.5 < angle_deg[row, col] < 135):
                        s = 1/abs(np.tan(angle[row, col]))
                        m1 = magnitude[row + 1, col] * (1 - s) + magnitude[row + 1, col - 1] * s
                        m2 = magnitude[row - 1, col] * (1 - s) + magnitude[row - 1, col + 1] * s
                    elif (135 < angle_deg[row, col] < 157.5):
                        s = abs(np.tan(angle[row, col]))
                        m1 = magnitude[row + 1, col] * (1 - s) + magnitude[row + 1, col - 1] * s
                        m2 = magnitude[row - 1, col] * (1 - s) + magnitude[row - 1, col + 1] * s

                if (magnitude[row, col] >= m1) and (magnitude[row, col] >= m2):
                    larger_magnitude[row, col] = magnitude[row, col]
                else:
                    larger_magnitude[row, col] = 0
            except IndexError:
                pass

    #larger_magnitude값을 0~255의 uint8로 변환
    larger_magnitude = (larger_magnitude / np.max(larger_magnitude) * 255).astype(np.uint8)

    return larger_magnitude

def double_thresholding(src):
    ############################################
    # TODO                                     #
    # double_thresholding 완성                 #
    # dst     : 진짜 edge만 남은 image         #
    ###########################################

    high_threshold_value, _ = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    row_threshold_value = high_threshold_value * 0.4

    (h, w) = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            if (src[row, col] > high_threshold_value):
                src[row,col] = 255
            elif (src[row, col] > row_threshold_value) and (src[row, col] <= high_threshold_value):
                src[row, col] = 100
            else:
                src[row, col] = 0

    for row in range(h):
        for col in range(w):
            if (src[row, col] == 100):
                hysteresis(src, row, col)

    return src

def hysteresis(src, row, col):
    if ((src[row + 1, col] == 255) or (src[row + 1, col - 1] == 255) or (src[row, col - 1] == 255) or (src[row + 1, col + 1] == 255)
        or (src[row, col + 1] == 255) or (src[row - 1, col - 1] == 255) or (src[row - 1, col] == 255) or (src[row - 1, col + 1] == 255)):
        src[row, col] = 255
        return 255
    elif (src[row - 1, col] == 100):#           # ↑ 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row - 1, col , row, col)
    elif (src[row - 1, col - 1] == 100):        # ↖ 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row - 1, col - 1 , row, col)
    elif (src[row, col - 1] == 100):            # <- 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row, col - 1 , row, col)
    elif (src[row + 1, col - 1] == 100):        # ↙ 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row + 1, col - 1 , row, col)
    elif (src[row + 1, col] == 100):            # ↓ 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row + 1, col , row, col)
    elif (src[row + 1, col + 1] == 100):        # ↘ 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row + 1, col + 1 , row, col)
    elif (src[row, col + 1] == 100):            # -> 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row, col + 1 , row, col)
    elif (src[row - 1, col + 1] == 100):        # ↗ 방향
        src[row, col] = 150
        src[row, col] = hysteresis(src, row - 1, col + 1 , row, col)
    else:
        src[row, col] = 0
        return 0
    return src[row, col]


# 여기
def my_canny_edge_detection(src, fsize=5, sigma=1, pad_type='zero'):
    #low-pass filter를 이용하여 blur효과
    #high-pass filter를 이용하여 edge 검출
    #gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma, pad_type)

    #magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    #non-maximum suppression 수행
    larger_magnitude = non_maximum_supression(magnitude, angle)

    #진짜 edge만 남김
    dst = double_thresholding(larger_magnitude)
    # dst = double_thresholding(src)

    return dst


if __name__ =='__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)

    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection_Gaussian', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()