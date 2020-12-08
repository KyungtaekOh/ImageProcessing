import cv2
import numpy as np
import my_padding as my_p

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    #mask의 크기
    (m_h, m_w) = mask.shape
    # mask 확인
    print('<mask>')
    print(mask)

    # 직접 구현한 my_padding 함수 이용
    pad_img = my_p.my_padding(src, (m_h//2, m_w//2), pad_type)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)

    return dst

def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    return sobel_x, sobel_y

if __name__ =='__main__':
    # src = cv2.imread('edge.PNG', cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    sobel_x, sobel_y = get_my_sobel()
    dst_x = my_filtering(src, sobel_x, 'repetition')
    dst_y = my_filtering(src, sobel_y, 'repetition')
    dst = np.sqrt(dst_x**2 + dst_y**2)

    ret, dst = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('original', src)
    cv2.imshow('sobel', dst/255)

    cv2.waitKey()
    cv2.destroyAllWindows()