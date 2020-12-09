import numpy as np
import cv2
import my_padding as my_p
def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    #2차 gaussian mask 생성
    gaus2D =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    #mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    print(gaus2D)
    return gaus2D

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape
    pad_img = my_p.my_padding(src, (m_h // 2, m_w // 2), pad_type)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
    return dst

def my_normalize(src):
    dst = src.copy()
    dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

def my_bilateral(src, msize, sigma, sigma_r, pad_type='zero'):
    ############################################
    # TODO                                     #
    # my_bilateral 함수 완성                   #
    # src : 원본 image                         #
    # msize : mask size                        #
    # sigma : sigma_x, sigma_y 값              #
    # sigma_r : sigma_r값                      #
    # pad_type : padding type                  #
    # dst : bilateral filtering 결과 image     #
    ############################################
    (a, b) = src.shape
    half_m = (int)(msize/2)
    pad_img  = my_p.my_padding(src, (half_m, half_m))
    (h, w) = pad_img.shape
    bilateral = np.zeros((msize, msize))
    img = np.zeros((msize, msize), dtype=np.uint8)
    dst = np.zeros((a, b), dtype=np.uint8)

    for i in range(2, h-2):
        for j in range(2, w-2):
            # bilateral mask 생성
            for row in range(-half_m, half_m+1):
                for col in range(-half_m, half_m+1):
                    bilateral[row+2, col+2] = np.exp(-(((i - i + row) ** 2) / (2 * (sigma ** 2)))
                                                     - (((j - j + col) ** 2) / (2 * (sigma ** 2)))) \
                                              * np.exp(-(((pad_img[i, j] - pad_img[i + row, j + col]) ** 2)
                                                         / (2 * (sigma_r ** 2))))
            sum_value = np.sum(bilateral)
            sum1_bilateral = bilateral / sum_value
            dst[i-2, j-2] = np.sum(pad_img[i-half_m:i+half_m+1, j-half_m:j+half_m+1]*sum1_bilateral)
            if i==53 and j == 123:
                mask = sum1_bilateral.copy()
                print(mask)
                mask_img = cv2.resize(mask,(200,200), interpolation=cv2.INTER_NEAREST)
                mask_img = my_normalize(mask_img)
                cv2.imshow('mask', mask_img)
                img = pad_img[i-half_m:i+half_m+1, j-half_m:j+half_m+1]
                img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
                img = my_normalize(img)
                cv2.imshow('mask img', img)
    dst = my_normalize(dst)

    return dst, mask

if __name__ == '__main__':
    src = cv2.imread('Penguins_noise.png', cv2.IMREAD_GRAYSCALE)
    dst, mask = my_bilateral(src, 5, 3, 30)

    gaus2D = my_get_Gaussian2D_mask(5 , sigma = 1)
    dst_gaus2D= my_filtering(src, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    cv2.imshow('original', src)
    cv2.imshow('my gaussian', dst_gaus2D)
    cv2.imshow('my bilateral', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

