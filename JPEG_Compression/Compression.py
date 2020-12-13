import numpy as np
import cv2
import my_JPEG_DCT2 as my_dct

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def my_JPEG_encoding(src, block_size=8):
    #####################################################
    # TODO                                              #
    # my_block_encoding 완성                            #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # zigzag_value : encoding 결과(zigzag까지)          #
    #####################################################

    (h, w) = src.shape
    # src_temp = src.copy()
    sub_image = src.copy() - 128            #sub_image
    dct = my_dct.my_DCT(sub_image)          #after DCT
    divide_Q = np.zeros((h, w)).astype(np.float)
    luminanace = Quantization_Luminance()
    (h_l, w_l) = luminanace.shape           #8,8   64, 64
    zigzag_value = list()
    index = 0
    block = np.zeros((block_size, block_size))

    for row in range(h // h_l):
        for col in range(w // w_l):
            divide_Q[row*h_l:(row+1)*h_l, col*w_l:(col+1)*w_l] = \
                dct[row*h_l:(row+1)*h_l, col*w_l:(col+1)*w_l] / luminanace
            temp = divide_Q[row*h_l:(row+1)*h_l, col*w_l:(col+1)*w_l]
            rounded = np.round(temp)
            # zigzag_index =
            zigzag_value.insert(index, scan_zigzag(rounded))
            index += 1
    zigzag_value.reverse()
    return zigzag_value

def scan_zigzag(src):
    (h, w) = src.shape
    temp = np.zeros(h*w)
    row, col, index, lastIndex = 0, 0, 0, 0
    dir = 0     # 0 : up, 1 : down , 10,11 : curv

    while(row < h and col < w):
        temp[index] = src[row, col]

        if (row == h-1 and (col % 2) == 1 and dir != 11 and dir != 10):     #오른쪽 인덱스
            col += 1
            dir = 11
        elif ((row % 2) == 0 and col == h-1 and dir != 11 and dir != 10):   #하단 인덱스
            row += 1
            dir = 10
        elif ((row % 2) == 0 and col == 0 and dir != 11 and dir != 10):     #상단 인덱스
            row += 1
            dir = 11
        elif (row == 0 and (col % 2) == 1 and dir != 11 and dir != 10):     #왼쪽 인덱스
            col += 1
            dir = 10
        else:
            if (dir == 10 or dir == 0):  # ↗ 방향
                row += 1
                col -= 1
                dir = 0
            elif (dir == 11 or dir == 1):  # ↙ 방향
                row -= 1
                col += 1
                dir = 1
        index += 1

    for i in range(63, -1, -1):
        if(temp[i] != 0):
            lastIndex = i + 1
            temp[i+1] = np.nan
            break
        if(temp[i] == 0 and i == 0):
            temp[i] = np.nan

    dst = np.zeros(lastIndex + 1)
    dst[0:lastIndex+1] = temp[0:lastIndex+1]
    return dst


def my_JPEG_decoding(zigzag_value, src, b_size=8): #src는 원본
    #####################################################
    # TODO                                              #
    # my_JPEG_decoding 완성                             #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # dst : decoding 결과 이미지                         #
    #####################################################
    (h, w) = src.shape      # 512, 512
    zigzag_temp = zigzag_value.copy()
    luminanace = Quantization_Luminance()
    length = len(zigzag_value)
    d_zigzag = np.zeros((h, w))

    for i in range(0, length):     # i = 0 ~ 63
        row = i % 64       # row = 0 ~ 7
        col = i // 64        # col = 0 ~ 7
        list_temp = zigzag_temp.pop()
        temp = decode_zigzag(b_size, list_temp)#row * b_size : (row + 1) * b_size
        d_zigzag[col * b_size: (col + 1) * b_size, row * b_size : (row + 1) * b_size] = \
            temp * luminanace
    d_idct = my_IDCT(d_zigzag, b_size)
    dst = d_idct + 128
    result = my_dct.my_normalize(dst)

    return result

def my_IDCT(src, n=8):
    (h, w) = src.shape
    F = src.copy()
    dst = np.zeros((h, w))

    for row in range(h // n):
        for col in range(w // n):
            temp = get_IDCT(F[row*n:(row+1)*n, col*n:(col+1)*n], n)
            dst[row*n:(row+1)*n, col*n:(col+1)*n] = temp

    return dst

def get_IDCT(F, n = 8):
    f = np.zeros((n,n))
    val = 0
    for u in range(n):
        for v in range(n):
            if (u == 0):
                C_u = 1 / np.sqrt(n)
            else:
                C_u = np.sqrt(2) / np.sqrt(n)

            if (v == 0):
                C_v = 1 / np.sqrt(n)
            else:
                C_v = np.sqrt(2) / np.sqrt(n)

            x, y = np.mgrid[0:n, 0:n]
            val = np.sum(F * C_u * C_v * np.cos(((2*x+1)*u*np.pi)/(2*n)) * np.cos(((2*y+1)*v*np.pi)/(2*n)))

            f[u, v] = val

    return f


def decode_zigzag(block_size, ary):
    row, col, index, nanCheck = 0, 0, 0, 0
    dir = 0
    dst = np.zeros((block_size, block_size))

    while (row < block_size and col < block_size):
        if (nanCheck == 0):
            if (np.isnan(ary[index])):
                break
            else:
                dst[row, col] = ary[index]
        else:
            dst[row, col] = 0
        if (row == block_size - 1 and (col % 2) == 1 and dir != 11 and dir != 10):  # 오른쪽 인덱스
            col += 1
            dir = 11
        elif ((row % 2) == 0 and col == block_size - 1 and dir != 11 and dir != 10):  # 하단 인덱스
            row += 1
            dir = 10
        elif ((row % 2) == 0 and col == 0 and dir != 11 and dir != 10):  # 상단 인덱스
            row += 1
            dir = 11
        elif (row == 0 and (col % 2) == 1 and dir != 11 and dir != 10):  # 왼쪽 인덱스
            col += 1
            dir = 10
        else:
            if (dir == 10 or dir == 0):  # ↗ 방향
                row += 1
                col -= 1
                dir = 0
            elif (dir == 11 or dir == 1):  # ↙ 방향
                row -= 1
                col += 1
                dir = 1
        index += 1

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    #이론ppt에 나와있는 배열 테스트하실분만 해보시면 될 것 같습니다.
    #참고로 이론 ppt에 나온 값하고 다르게 나옵니다.
    #(예를 들어 인덱스가 [5,5]인 68을 예로 들면 68 - 128을 하면 -60이 나와야 하는데
    #이론ppt에는 -65로 값이 잘못나와있습니다. 뭔가 값이 조금씩 다르게 나옵니다. 그러니 참고용으로만 사용해주세요)
    
    # src = np.array(
    #     [[52, 55, 61, 66, 70, 61, 64, 73],
    #      [63, 59, 66, 90, 109, 85, 69, 72],
    #      [62, 59, 68, 113, 144, 104, 66, 73],
    #      [63, 58, 71, 122, 154, 106, 70, 69],
    #      [67, 61, 68, 104, 126, 88, 68, 70],
    #      [79, 65, 60, 70, 77, 68, 58, 75],
    #      [85, 71, 64, 59, 55, 61, 65, 83],
    #      [87, 79, 69, 68, 65, 76, 78, 94]])


    src = src.astype(np.float)
    zigzag_value = my_JPEG_encoding(src)
    print(zigzag_value[:10])

    dst = my_JPEG_decoding(zigzag_value, src)
    src = src.astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('result', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


