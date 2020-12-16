import numpy as np
import cv2
import my_padding as my_p

def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    #2차 gaussian mask 생성
    gaus2D =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))

    #mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D

def my_filtering(src, mask, pad_type = 'zero'):
    (h, w) = src.shape

    #mask의 크기
    (m_h, m_w) = mask.shape

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_p.my_padding(src, (m_h//2, m_w//2), pad_type)

    dst = np.zeros((h, w))

    #시간을 확인하려면 4중 for문을 이용해야함
    for row in range(h):
        for col in range(w):
            sum = 0
            for m_row in range(m_h):
                for m_col in range(m_w):
                    sum += pad_img[row + m_row, col+m_col] * mask[m_row, m_col]
            dst[row, col] = sum

    #4중 for문 시간이 오래걸릴 경우 해당 코드 사용(시간을 확인하려면 해당 코드를 사용하면 안됨)
    # for row in range(h):
    #     for col in range(w):
    #         dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)

    return dst

def my_laplacian_pyramids(src, repeat, gap=2, msize=3, sigma=1, pad_type='zero'):
    dsts_down = []  #결과를 저장하기 위한선언
    dsts_up = []
    residuals = []
    dsts_down.append(src.copy())    #copy 깊은복사 사용하는거 추천
    gaus2D = my_get_Gaussian2D_mask(msize, sigma)
    for i in range(repeat):
        dst, res = my_laplacian_downsampling(dsts_down[i], gap, gaus2D, pad_type)
        dsts_down.append(dst)
        residuals.append(res)

    dsts_up.append(dsts_down[repeat])
    for i in range(repeat):                                      #-는 뒤에서부터
        dsts_up.append(my_laplacian_upsampling(dsts_up[i], gap, residuals[-(i+1)]))

    return dsts_down, dsts_up, residuals

def my_laplacian_downsampling(src, gap, mask, pad_type ='zero'):
    (h, w) = src.shape
    blur_img = my_filtering(src, mask, pad_type)
    res = src - blur_img

    dst = np.zeros((h//gap, w//gap))
    (h_dst, w_dst) = dst.shape
    for row in range(h_dst):
        for col in range(w_dst):
            dst[row, col] = blur_img[row*gap, col*gap]

    dst = (dst+0.5).astype(np.uint8)
    return dst, res

def my_laplacian_upsampling(src, gap, residual):
    (h, w) = src.shape

    dst = np.zeros((h//gap, w//gap))
    (h_dst, w_dst) = dst.shape
    for row in range(h_dst):
        for col in range(w_dst):
            intensity = src[row//gap, col//gap]+residual[row,col]
            intensity = max(min(intensity, 255), 0) #overflow나 underflow방지
            dst[row, col] = intensity

    dst = (dst+0.5).astype(np.uint8)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    dsts_lapla_down, dsts_lapla_up, res = my_laplacian_pyramids(src, 2, msize=7, sigma=5, pad_type='repetition')
    for i in range(len(dsts_lapla_down)):
        cv2.imshow('gaussian dst%d downsampling'%i, dsts_lapla_down[i])

    for i in range(len(dsts_lapla_up)):
        cv2.imshow('gaussian dsts%d upsampling'%i, dsts_lapla_up[i])

    for i in range(len(res)):
        cv2.imshow('res%d'%i, res[i]/255)

    cv2.waitKey()
    cv2.destroyAllWindows()

