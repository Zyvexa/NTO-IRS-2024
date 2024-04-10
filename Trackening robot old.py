import cv2
import numpy as np
import math
from math import sqrt

# cam_left = cv2.VideoCapture('http://student:nto2024@10.128.73.31/mjpg/video.mjpg')
# cam_right = cv2.VideoCapture('http://student:nto2024@10.128.73.38/mjpg/video.mjpg')
cam_usb = cv2.VideoCapture(1)
cam_usb.set(cv2.CAP_PROP_AUTO_WB, 0.0) 
# cam_usb.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# cam_usb.set(cv2.CAP_PROP_FOCUS, 0)

CRX, CRY = 0, 0  
NX, NY = 0, 0

camera_matrix_left = np.array([[1.87037597e+03, 0.00000000e+00, 5.62163412e+02],
       [0.00000000e+00, 1.86630329e+03, 3.48379060e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coefs_left = np. array([-1.31752408, -1.71491928,  0.04980193, -0.06095647, 19.891681  ])

camera_matrix_right = np.array([[822.80933559,   0.        , 656.1610982 ],
       [  0.        , 823.59477124, 423.88522274],
       [  0.        ,   0.        ,   1.        ]])
dist_coefs_right = np. array([-0.24993429,  0.10347935, -0.00134318, -0.00059863, -0.03127951])

camera_matrix_usb = np.array([[744.78560704,   0.        , 416.79106153],
       [  0.        , 746.17464309, 310.35201284],
       [  0.        ,   0.        ,   1.        ]])
dist_coefs_usb = np. array([-0.01179911,  0.19509182,  0.00176673,  0.00154678, -0.42737765])
# Функции Обработки
def blue_filter(image):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определение нижнего и верхнего порогов для синего цвета в HSV
    lower_blue = np.array([100, 50, 50])  # Нижний порог для оттенков синего
    upper_blue = np.array([140, 255, 255])  # Верхний порог для оттенков синего

    # Создание маски для синего цвета
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Применение маски к изображению
    blue_result = cv2.bitwise_and(image, image, mask=blue_mask)
    return blue_result


def robot_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([10, 90, 90])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 90, 90])
    upper_red = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    filtered = cv2.bitwise_and(image, image, mask=mask)
    return filtered

def robot_nap(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([50, 50, 50])  
    upper_green = np.array([95, 255, 255])
    mask3 = cv2.inRange(hsv, lower_green, upper_green)

    lower_red = np.array([0, 90, 90])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 90, 90])
    upper_red = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2 + mask3

    filtered = cv2.bitwise_and(image, image, mask=mask)
    return filtered

def fish_eye(img,cam):
    h,  w = img.shape[:2]
    if cam == "left":
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix_left, dist_coefs_left, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix_left, dist_coefs_left, None, newcameramtx)
        
    elif cam == "right":
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix_right, dist_coefs_right, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix_right, dist_coefs_right, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]                   
    elif cam == "usb":
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix_usb, dist_coefs_usb, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix_usb, dist_coefs_usb, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def image_effects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    edges = cv2.Canny(blurred, 150, 60)
    dil = cv2.dilate(edges, kernel, iterations = 1)
    return dil

kernel = np.ones((5,5))

global_pl_y = 0
gl_pl_x = 0

while True:
    # ret, frame_l = cam_left.read()
    # ret, frame_r = cam_right.read()
    # ret, image = cam_usb.read()
    image = cv2.imread("n13.png")
    
    frame = image                     #используем камеру 
    frame = fish_eye(frame, "usb")      #устраняем 
    
    #ПОИСК РОБОТА
    cont_raw = cv2.findContours(image_effects(robot_red(frame)), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, h = cont_raw[0], cont_raw[1]

    good_1 = []
    for contur in contours:
        if cv2.contourArea(contur) > 50:
            good_1.append(contur)

    try:
        M1 = cv2.moments(good_1[0])
        M2 = cv2.moments(good_1[1])

        if M1['m00'] != 0 and M2['m00'] != 0:
            cx1 = int(M1['m10'] / M1['m00'])
            cx2 = int(M2['m10'] / M2['m00'])
            cx = int((cx1+cx2)/2)
            CRX = cx

            cy1 = int(M1['m01'] / M1['m00'])
            cy2 = int(M2['m01'] / M2['m00'])
            cy = int((cy1+cy2)/2)
            CRY = cy

            
            # cv2.rectangle(frame,(cx-32,cy+30),(cx+50,cy-50),(0,0,255),1)
            cropped_image = frame[cy-30:cy+30, cx-30:cx+30]
            cropped_image_copy = frame[cy-30:cy+30, cx-30:cx+30].copy()
            global_pl_y = cy - 30
            gl_pl_x = cx - 30
    except: pass

    
    # ПОИСК НАПРАВЛЕНИЯ
    try:
        cont_raw = cv2.findContours(image_effects(robot_nap(cropped_image)), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours, h = cont_raw[0], cont_raw[1]

        # исключение лишних контуров
        good = []
        for contur in contours:
            if cv2.contourArea(contur) > 50:
                good.append(contur)
        print(len(good))
        try:
            M1 = cv2.moments(good[0])
            M2 = cv2.moments(good[1])
            M3 = cv2.moments(good[2])
            M4 = cv2.moments(good[3])

            cv2.drawContours(cropped_image, good[0], -1, (255,0,0), 2)
            cv2.drawContours(cropped_image, good[1], -1, (0,255,0), 2)
            cv2.drawContours(cropped_image, good[2], -1, (0,0,255), 2)
            cv2.drawContours(cropped_image, good[3], -1, (255,255,255), 2)
            
            # cv2.imshow('frame', frame)
            # cv2.imshow('framesdf2', cropped_image_copy)
            # cv2.waitKey(0)

            if M1['m00'] != 0 and M2['m00'] != 0 and M3['m00'] != 0 and M4['m00'] != 0:
                cx1 = int(M1['m10'] / M1['m00'])
                cy1 = int(M1['m01'] / M1['m00'])
                cv2.circle(cropped_image, (cx1,cy1), 1, (255,0,0), 1)

                cx2 = int(M2['m10'] / M2['m00'])
                cy2 = int(M2['m01'] / M2['m00'])
                cv2.circle(cropped_image, (cx2,cy2), 1, (0,255,0), 1)

                cx3 = int(M3['m10'] / M3['m00'])
                cy3 = int(M3['m01'] / M3['m00'])
                cv2.circle(cropped_image, (cx3,cy3), 1, (0,0,255), 1)

                cx4 = int(M4['m10'] / M4['m00'])
                cy4 = int(M4['m01'] / M4['m00'])
                cv2.circle(cropped_image, (cx4,cy4), 1, (255,255,255), 1)
                
                # cv2.imshow('frame', frame)
                # cv2.imshow('framesdf2', frame)
                # cv2.waitKey(0)

                r_1_2, r_1_3, r_1_4= int(sqrt((cx1-cx2)**2 + (cy1-cy2)**2)), int(sqrt((cx1-cx3)**2 + (cy1-cy3)**2)), int(sqrt((cx1-cx4)**2 + (cy1-cy4)**2))
                r_2_3, r_2_4 = int(sqrt((cx2-cx3)**2 + (cy2-cy3)**2)), int(sqrt((cx2-cx4)**2 + (cy2-cy4)**2))
                r_3_4 = int(sqrt((cx3-cx4)**2 + (cy3-cy4)**2))

                minotr = min(r_1_2, r_1_3,r_1_4,r_2_3,r_2_4,r_3_4)
                x_vr = 0
                y_vr = 0
                if minotr == r_1_2: 
                    x_vr = int((cx1+cx2)/2)
                    y_vr = int((cy1+cy2)/2)
                    # cv2.circle(cropped_image, (int((cx1+cx2)/2), int((cy1+cy2)/2)), 4, (0,255,255), 2)
                elif minotr == r_1_3:
                    x_vr = int((cx1+cx3)/2)
                    y_vr = int((cy1+cy3)/2)
                    # cv2.circle(cropped_image, (int((cx1+cx3)/2), int((cy1+cy3)/2)), 4, (0,255,255), 2)
                elif minotr == r_1_4: 
                    x_vr = int((cx1+cx4)/2)
                    y_vr = int((cy1+cy4)/2)
                    # cv2.circle(cropped_image, (int((cx1+cx4)/2), int((cy1+cy4)/2)), 4, (0,255,255), 2)
                elif minotr == r_2_3: 
                    x_vr = int((cx2+cx3)/2)
                    y_vr = int((cy2+cy3)/2)
                    # cv2.circle(cropped_image, (int((cx2+cx3)/2), int((cy2+cy3)/2)), 4, (0,255,255), 2)
                elif minotr == r_2_4: 
                    x_vr = int((cx2+cx4)/2)
                    y_vr = int((cy2+cy4)/2)
                    # cv2.circle(cropped_image, (int((cx2+cx4)/2), int((cy2+cy4)/2)), 4, (0,255,255), 2)
                elif minotr == r_3_4: 
                    x_vr = int((cx3+cx4)/2)
                    y_vr = int((cy3+cy4)/2)
                    # cv2.circle(cropped_image, (int((cx3+cx4)/2), int((cy3+cy4)/2)), 4, (0,255,255), 2)
                cv2.circle(cropped_image, (x_vr, y_vr), 4, (0,255,255), 2)
                NX, NY = x_vr + gl_pl_x, y_vr  + global_pl_y
                
                cv2.circle(frame, (525, 235), 5, (255,0,255), -1)
                cv2.circle(frame, (530, 350), 5, (255,0,255), -1)
                
                print(CRX, CRY)
        except: pass
        
        cv2.line(frame,(NX, NY),(CRX, CRY),(255,0,0),2)
        cv2.line(frame,(CRX, CRY),(699,447),(255,0,0),2)

        cv2.namedWindow('VideoCrop', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('VideoCrop', 500, 500)  # Установка размера окна
        cv2.imshow('VideoCrop', cropped_image)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except: 
        print("nan")
        break





# def angle_between_points(point1, point2, point3):
#     x1, y1 = point1
#     x2, y2 = point2
#     x3, y3 = point3

#     # Вычисляем векторы между точками
#     vector1 = (x1 - x2, y1 - y2)
#     vector2 = (x3 - x2, y3 - y2)

#     # Вычисляем скалярное произведение векторов
#     dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

#     # Вычисляем длины векторов
#     magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
#     magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

#     # Вычисляем угол между векторами
#     cos_angle = dot_product / (magnitude1 * magnitude2)
#     angle_radians = math.acos(cos_angle)
#     angle_degrees = math.degrees(angle_radians)

#     return angle_degrees

# angle = angle_between_points((NX, NY), (CRX, CRY), (699,447))
# print("Угол между точками:", angle)    
