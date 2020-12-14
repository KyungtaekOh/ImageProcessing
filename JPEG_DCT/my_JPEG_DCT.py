import cv2
import numpy as np

def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

def my_DCT(src, n=8):
    ###############################
    # TODO                        #
    # my_DCT 완성                 #
    # src : input image           #
    # n : block size              #
    ###############################

    (h, w) = src.shape
    dst = np.zeros((h, w)).astype(np.float)
    dct = np.zeros((n, n)).astype(np.float)
    temp = np.zeros((n, n)).astype(np.float)

    sigma = 0
    u_value = np.sqrt(2) / np.sqrt(n)
    v_value = np.sqrt(2) / np.sqrt(n)

    for row_src in range(h // n):       #================== index src ====================
        for col_src in range(w // n):
            dct = src[row_src*n : (row_src+1)*n, col_src*n :(col_src+1)*n]
            for u in range(n):          #================== index u,v ====================
                for v in range(n):

                    for row_sigma in range(n):      #============== index sigma ================
                        for col_sigma in range(n):
                            sigma += (dct[row_sigma, col_sigma] *
                                      (np.cos((((2 * row_sigma) + 1) * u * np.pi) / (2 * n))) *
                                      (np.cos((((2 * col_sigma) + 1) * v * np.pi) / (2 * n))))

                    if (u == 0):
                        temp[u, v] = sigma * (1/np.sqrt(n)) * v_value
                    elif (v == 0):
                        temp[u, v] = sigma * u_value * (1/np.sqrt(n))
                    elif (u == 0 and v == 0):
                        temp[u, v] = sigma * (1/np.sqrt(n)) * (1/np.sqrt(n))
                    else:
                        temp[u, v] = sigma * u_value * v_value
                    sigma = 0

            dst[row_src * n : (row_src + 1)*n, col_src * n : (col_src + 1) * n] = temp

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_DCT(src, 5)

    dst = my_normalize(dst)
    cv2.imshow('my DCT', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


