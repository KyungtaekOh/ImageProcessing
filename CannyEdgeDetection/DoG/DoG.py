import cv2
import numpy as np
import my_padding as my_p

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    #mask의 크기
    (m_h, m_w) = mask.shape
    # mask 확인

    # 직접 구현한 my_padding 함수 이용
    pad_img = my_p.my_padding(src, (m_h//2, m_w//2), pad_type)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)

    return dst

def get_my_DoG(msize, sigma=1):
    y, x = np.mgrid[-(msize//2):(msize//2)+1, -(msize//2):(msize//2)+1]

    DoG_x = (-x / sigma**2) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    DoG_y = (-y / sigma**2) * np.exp(-(x**2 + y**2)/(2*sigma**2))

    return DoG_x, DoG_y

if __name__ =='__main__':
    # src = cv2.imread('edge.PNG', cv2.IMREAD_GRAYSCALE)
    # src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('double_threshold_test_img.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_my_DoG(5)

    dst_x = my_filtering(src, DoG_x, 'repetition')
    dst_y = my_filtering(src, DoG_y, 'repetition')
    # dst = np.sqrt(dst_x ** 2 + dst_y ** 2)
    dst_temp = np.sqrt(dst_x**2 + dst_y**2)
    dst = (dst_temp).astype(np.uint8)
    # cv2.imshow('x', dst_x)
    # cv2.imshow('y', dst_y)
    cv2.imshow('DoG filter', dst/np.max(dst))

    cv2.waitKey()
    cv2.destroyAllWindows()