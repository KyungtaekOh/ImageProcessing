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

    h_pad = h + (n - h%n)
    w_pad = w + (n - w%n)

    pad_img = np.zeros((h_pad, w_pad))
    pad_img[:h, :w] = src.copy()
    dst = np.zeros((h_pad, w_pad))

    for row in range(h_pad // n):
        for col in range(w_pad // n):
            dst[row*n:(row+1)*n, col*n:(col+1)*n] = \
                get_DCT(pad_img[row*n:(row+1)*n, col*n:(col+1)*n])

    return dst[:h, :w]

def get_DCT(f, n = 8):
    F = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            x, y = np.mgrid[0:n, 0:n]
            val = np.sum(f*np.cos(((2*x+1)*u*np.pi)/(2*n)) * np.cos(((2*y+1)*v*np.pi)/(2*n)))

            if(u == 0):
                C_u = 1/np.sqrt(n)
            else:
                C_u = np.sqrt(2) / np.sqrt(n)
            if (v == 0):
                C_v = 1 / np.sqrt(n)
            else:
                C_v = np.sqrt(2) / np.sqrt(n)

            F[u, v] = C_u * C_v * val

    return F

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_DCT(src, 8)

    dst = my_normalize(dst)
    cv2.imshow('my DCT', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


