import cv2
import numpy as np

r90_idx = [6, 3, 0, 7, 4, 1, 8, 5, 2]
r180_idx = [8, 7, 6, 5, 4, 3, 2, 1, 0]
r270_idx = [2, 5, 8, 1, 4, 7, 0, 3, 6]

aruco_3x3_dict = {
    1: [0, 0, 0, 0, 0, 0, 0, 0, 1],
    2: [0, 0, 0, 0, 0, 0, 0, 1, 0],
    3: [0, 0, 0, 0, 0, 0, 0, 1, 1],
    5: [0, 0, 0, 0, 0, 0, 1, 0, 1],
    6: [0, 0, 0, 0, 0, 0, 1, 1, 0],
    7: [0, 0, 0, 0, 0, 0, 1, 1, 1],
    10: [0, 0, 0, 0, 0, 1, 0, 1, 0],
    11: [0, 0, 0, 0, 0, 1, 0, 1, 1],
    12: [0, 0, 0, 0, 0, 1, 1, 0, 0],
    13: [0, 0, 0, 0, 0, 1, 1, 0, 1],
    14: [0, 0, 0, 0, 0, 1, 1, 1, 0],
    15: [0, 0, 0, 0, 0, 1, 1, 1, 1],
    17: [0, 0, 0, 0, 1, 0, 0, 0, 1],
    18: [0, 0, 0, 0, 1, 0, 0, 1, 0],
    19: [0, 0, 0, 0, 1, 0, 0, 1, 1],
    21: [0, 0, 0, 0, 1, 0, 1, 0, 1],
    22: [0, 0, 0, 0, 1, 0, 1, 1, 0],
    23: [0, 0, 0, 0, 1, 0, 1, 1, 1],
    26: [0, 0, 0, 0, 1, 1, 0, 1, 0],
    27: [0, 0, 0, 0, 1, 1, 0, 1, 1],
    28: [0, 0, 0, 0, 1, 1, 1, 0, 0],
    29: [0, 0, 0, 0, 1, 1, 1, 0, 1],
    30: [0, 0, 0, 0, 1, 1, 1, 1, 0],
    31: [0, 0, 0, 0, 1, 1, 1, 1, 1],
    33: [0, 0, 0, 1, 0, 0, 0, 0, 1],
    35: [0, 0, 0, 1, 0, 0, 0, 1, 1],
    37: [0, 0, 0, 1, 0, 0, 1, 0, 1],
    39: [0, 0, 0, 1, 0, 0, 1, 1, 1],
    41: [0, 0, 0, 1, 0, 1, 0, 0, 1],
    42: [0, 0, 0, 1, 0, 1, 0, 1, 0],
    43: [0, 0, 0, 1, 0, 1, 0, 1, 1],
    44: [0, 0, 0, 1, 0, 1, 1, 0, 0],
    45: [0, 0, 0, 1, 0, 1, 1, 0, 1],
    46: [0, 0, 0, 1, 0, 1, 1, 1, 0],
    47: [0, 0, 0, 1, 0, 1, 1, 1, 1],
    49: [0, 0, 0, 1, 1, 0, 0, 0, 1],
    51: [0, 0, 0, 1, 1, 0, 0, 1, 1],
    53: [0, 0, 0, 1, 1, 0, 1, 0, 1],
    55: [0, 0, 0, 1, 1, 0, 1, 1, 1],
    57: [0, 0, 0, 1, 1, 1, 0, 0, 1],
    58: [0, 0, 0, 1, 1, 1, 0, 1, 0],
    59: [0, 0, 0, 1, 1, 1, 0, 1, 1],
    60: [0, 0, 0, 1, 1, 1, 1, 0, 0],
    61: [0, 0, 0, 1, 1, 1, 1, 0, 1],
    62: [0, 0, 0, 1, 1, 1, 1, 1, 0],
    63: [0, 0, 0, 1, 1, 1, 1, 1, 1],
    69: [0, 0, 1, 0, 0, 0, 1, 0, 1],
    70: [0, 0, 1, 0, 0, 0, 1, 1, 0],
    71: [0, 0, 1, 0, 0, 0, 1, 1, 1],
    76: [0, 0, 1, 0, 0, 1, 1, 0, 0],
    77: [0, 0, 1, 0, 0, 1, 1, 0, 1],
    78: [0, 0, 1, 0, 0, 1, 1, 1, 0],
    79: [0, 0, 1, 0, 0, 1, 1, 1, 1],
    85: [0, 0, 1, 0, 1, 0, 1, 0, 1],
    86: [0, 0, 1, 0, 1, 0, 1, 1, 0],
    87: [0, 0, 1, 0, 1, 0, 1, 1, 1],
    92: [0, 0, 1, 0, 1, 1, 1, 0, 0],
    93: [0, 0, 1, 0, 1, 1, 1, 0, 1],
    94: [0, 0, 1, 0, 1, 1, 1, 1, 0],
    95: [0, 0, 1, 0, 1, 1, 1, 1, 1],
    97: [0, 0, 1, 1, 0, 0, 0, 0, 1],
    98: [0, 0, 1, 1, 0, 0, 0, 1, 0],
    99: [0, 0, 1, 1, 0, 0, 0, 1, 1],
    101: [0, 0, 1, 1, 0, 0, 1, 0, 1],
    102: [0, 0, 1, 1, 0, 0, 1, 1, 0],
    103: [0, 0, 1, 1, 0, 0, 1, 1, 1],
    105: [0, 0, 1, 1, 0, 1, 0, 0, 1],
    106: [0, 0, 1, 1, 0, 1, 0, 1, 0],
    107: [0, 0, 1, 1, 0, 1, 0, 1, 1],
    109: [0, 0, 1, 1, 0, 1, 1, 0, 1],
    110: [0, 0, 1, 1, 0, 1, 1, 1, 0],
    111: [0, 0, 1, 1, 0, 1, 1, 1, 1],
    113: [0, 0, 1, 1, 1, 0, 0, 0, 1],
    114: [0, 0, 1, 1, 1, 0, 0, 1, 0],
    115: [0, 0, 1, 1, 1, 0, 0, 1, 1],
    117: [0, 0, 1, 1, 1, 0, 1, 0, 1],
    118: [0, 0, 1, 1, 1, 0, 1, 1, 0],
    119: [0, 0, 1, 1, 1, 0, 1, 1, 1],
    121: [0, 0, 1, 1, 1, 1, 0, 0, 1],
    122: [0, 0, 1, 1, 1, 1, 0, 1, 0],
    123: [0, 0, 1, 1, 1, 1, 0, 1, 1],
    125: [0, 0, 1, 1, 1, 1, 1, 0, 1],
    126: [0, 0, 1, 1, 1, 1, 1, 1, 0],
    127: [0, 0, 1, 1, 1, 1, 1, 1, 1],
    141: [0, 1, 0, 0, 0, 1, 1, 0, 1],
    142: [0, 1, 0, 0, 0, 1, 1, 1, 0],
    143: [0, 1, 0, 0, 0, 1, 1, 1, 1],
    157: [0, 1, 0, 0, 1, 1, 1, 0, 1],
    158: [0, 1, 0, 0, 1, 1, 1, 1, 0],
    159: [0, 1, 0, 0, 1, 1, 1, 1, 1],
    171: [0, 1, 0, 1, 0, 1, 0, 1, 1],
    173: [0, 1, 0, 1, 0, 1, 1, 0, 1],
    175: [0, 1, 0, 1, 0, 1, 1, 1, 1],
    187: [0, 1, 0, 1, 1, 1, 0, 1, 1],
    189: [0, 1, 0, 1, 1, 1, 1, 0, 1],
    191: [0, 1, 0, 1, 1, 1, 1, 1, 1],
    197: [0, 1, 1, 0, 0, 0, 1, 0, 1],
    199: [0, 1, 1, 0, 0, 0, 1, 1, 1],
    205: [0, 1, 1, 0, 0, 1, 1, 0, 1],
    206: [0, 1, 1, 0, 0, 1, 1, 1, 0],
    207: [0, 1, 1, 0, 0, 1, 1, 1, 1],
    213: [0, 1, 1, 0, 1, 0, 1, 0, 1],
    215: [0, 1, 1, 0, 1, 0, 1, 1, 1],
    221: [0, 1, 1, 0, 1, 1, 1, 0, 1],
    222: [0, 1, 1, 0, 1, 1, 1, 1, 0],
    223: [0, 1, 1, 0, 1, 1, 1, 1, 1],
    229: [0, 1, 1, 1, 0, 0, 1, 0, 1],
    231: [0, 1, 1, 1, 0, 0, 1, 1, 1],
    237: [0, 1, 1, 1, 0, 1, 1, 0, 1],
    239: [0, 1, 1, 1, 0, 1, 1, 1, 1],
    245: [0, 1, 1, 1, 1, 0, 1, 0, 1],
    247: [0, 1, 1, 1, 1, 0, 1, 1, 1],
    253: [0, 1, 1, 1, 1, 1, 1, 0, 1],
    255: [0, 1, 1, 1, 1, 1, 1, 1, 1],
    327: [1, 0, 1, 0, 0, 0, 1, 1, 1],
    335: [1, 0, 1, 0, 0, 1, 1, 1, 1],
    343: [1, 0, 1, 0, 1, 0, 1, 1, 1],
    351: [1, 0, 1, 0, 1, 1, 1, 1, 1],
    367: [1, 0, 1, 1, 0, 1, 1, 1, 1],
    383: [1, 0, 1, 1, 1, 1, 1, 1, 1]
}

dict_of_aruco = {}

def filt(img):
    originalImage = img

    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.blur(grayImage, (5, 5))

    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 54 , 255, cv2.THRESH_BINARY)

    # cv2.imshow('Black white image', blackAndWhiteImage)
    # cv2.imshow('Original image', originalImage)
    # cv2.imshow('Gray image', grayImage)

    return blackAndWhiteImage


def bin2dec(binarray):
    marker_id = 0
    n = len(binarray)
    for i in range(n - 1, -1, -1):
        marker_id += binarray[i] * 2 ** (n - i - 1)
    if marker_id in aruco_3x3_dict.keys():
        return marker_id
    else:
        return -1


def update_binarray(binarray, order):
    new_binarray = []
    for i in order:
        new_binarray.append(binarray[i])
    return new_binarray


def get_mask_binarray(mask):
    N = 5
    cell_width = mask.shape[1] / N
    cell_height = mask.shape[0] / N
    cell_cx = cell_width / 2
    cell_cy = cell_height / 2
    binarray = []
    is_boder = True
    for i in range(N):
        for j in range(N):
            y = round(i * cell_width + cell_cx)
            x = round(j * cell_height + cell_cy)
            color = round(np.average(mask[y - 1:y + 1, x - 1:x + 1]) / 255)

            if (i == 0 or i == N - 1) or (j == 0 or j == N - 1):
                is_boder = is_boder and (color == 0)
                continue
            binarray.append(color)
    if not is_boder:
        return []
    return binarray


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def fish_eye(img):
    h, w = img.shape[:2]
    camera_matrix_usb = np.array([[744.78560704,   0.        , 416.79106153],
       [  0.        , 746.17464309, 310.35201284],
       [  0.        ,   0.        ,   1.        ]])
    dist_coefs_usb = np. array([-0.01179911,  0.19509182,  0.00176673,  0.00154678, -0.42737765])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_usb, dist_coefs_usb, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix_usb, dist_coefs_usb, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # numpy_horizontal_concat = np.concatenate((img, dst), axis=1)
    # cv2.imshow('N', numpy_horizontal_concat)

    cv2.waitKey()
    return dst


def get_aruco_list(img):
    global dict_of_aruco
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img', imgray)
    # cv2.waitKey()    
    img = fish_eye(img)
    # img = cv2.resize(img,(1720,1250),interpolation = cv2.INTER_AREA)
    img = cv2.resize(img, None, fx=2.1, fy=2.14)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 181, 33)

    thresh = filt(img)
    # cv2.imshow('img', thresh) 
    # cv2.waitKey()

    contours, hi = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):

        if cv2.contourArea(cnt) < 40:
            continue
        if hi[0][i][2] == -1 or hi[0][i][3] == -1:
            continue
        c, dim, ang = cv2.minAreaRect(cnt)
        box = cv2.boxPoints((c, dim, ang))
        box = np.intp(box)

        x, y, w, h = cv2.boundingRect(cnt)

        if max(dim) < 40:
            continue

        # print(round(min(dim)), 'min')
        # print(round(max(dim)), 'max')
        f1 = round(min(dim))
        f2 = round(max(dim))

        if f1 / f2 < 0.6:
            continue
        # Найдем центр прямоугольника
        center_x = x + w // 2
        center_y = y + h // 2

        # Нарисуем центр прямоугольника
        cv2.circle(img, (center_x, center_y), 11, (0, 0, 255), -1)
        margin = 10
        x, y, w, h = cv2.boundingRect(cnt)

        mask = thresh[y - margin:y + h + margin, x - margin:x + w + margin]

        M = cv2.getRotationMatrix2D((mask.shape[0] / 2, mask.shape[1] / 2), ang, 1)
        mask = cv2.warpAffine(mask, M, (mask.shape[0], mask.shape[1]))

        clip = [round((mask.shape[0] - dim[1]) / 2), round((mask.shape[1] - dim[0]) / 2)]
        mask = mask[clip[0]:-clip[0], clip[1]:-clip[1]]
        # print(mask,'s')
        binarray = get_mask_binarray(mask)
        if len(binarray) == 0:
            continue
        # decoding
        marker_id = -1
        marker_ang = 0
        idx = [bin2dec(binarray), bin2dec(update_binarray(binarray, r90_idx)),
               bin2dec(update_binarray(binarray, r180_idx)),
               bin2dec(update_binarray(binarray, r270_idx))]
        for i, id in enumerate(idx):
            if id > -1:
                marker_id = id
                marker_ang = i * 90 - ang
                break

        if marker_id == -1:
            continue
        x_c = int(center_x / 2.1)
        y_c = int(center_y / 2.12)
        angular = int(marker_ang)
        # print(marker_id, ":[", x_c, ',', y_c, ',', angular, "],")
        dict_of_aruco[marker_id] = [x_c, y_c, angular]
        cv2.drawContours(img, [box], 0, (0, 0, 255), 6)
        cv2.putText(img, f'{x_c}, {y_c}; {marker_id}; {angular}', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    # cv2.imshow( 'img', img)
    # cv2.waitKey()
    return dict_of_aruco 
    # print(dict_of_aruco)
        # cv2.drawContours(thresh, lcnt], -1,  150, 3) cv2.imshow('img', mask) cv2.waitKey()
    


if __name__ == '__main__':
    cam_usb = cv2.VideoCapture(0)
    cam_usb.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    _, image = cam_usb.read()
    # image = cv2.imread("photos/n13.png")
    print(get_aruco_list(image))
