import cv2
import numpy as np
import math
from math import sqrt

cam_usb = cv2.VideoCapture(1)
cam_usb.set(cv2.CAP_PROP_AUTO_WB, 0.0) 
# cam_usb.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# cam_usb.set(cv2.CAP_PROP_FOCUS, 0)

CRX, CRY = 0, 0  
NX, NY = 0, 0

camera_matrix_usb = np.array([[744.78560704,   0.        , 416.79106153],
       [  0.        , 746.17464309, 310.35201284],
       [  0.        ,   0.        ,   1.        ]])
dist_coefs_usb = np. array([-0.01179911,  0.19509182,  0.00176673,  0.00154678, -0.42737765])

# Функции Обработки

def robot_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 90, 90])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([150, 90, 90])
    upper_red = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    filtered = cv2.bitwise_and(image, image, mask=mask)
    return filtered

def robot_nap(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([45, 45, 45])  
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

    if cam == "usb":
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

kernel = np.ones((4,4))

while True:
    # ret, image = cam_usb.read()
    image = cv2.imread("photos/n13.png")

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

            cropped_image = frame[cy-30:cy+30, cx-30:cx+30]
            # cropped_image_copy = frame[cy-30:cy+30, cx-30:cx+30].copy()
            global_pl_y = cy - 30
            gl_pl_x = cx - 30
    except: pass

    # ПОИСК НАПРАВЛЕНИЯ
    try:
        cont_raw = cv2.findContours(image_effects(robot_nap(cropped_image)), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours, h = cont_raw[0], cont_raw[1]

        # cv2.imshow('VideoCropsdm.wdmw.', image_effects(robot_nap(cropped_image)))
        # cv2.waitKey(0)
        # print(len(contours))

        # исключение лишних контуров
        good = []
        for contur in contours:
            if cv2.contourArea(contur) > 50:
                good.append(contur)
        # print(len(good))

        try:
            M1 = cv2.moments(good[0])
            M2 = cv2.moments(good[1])
            M3 = cv2.moments(good[2])
            M4 = cv2.moments(good[3])

            cv2.drawContours(cropped_image, good[0], -1, (255,0,0), 2)
            cv2.drawContours(cropped_image, good[1], -1, (0,255,0), 2)
            cv2.drawContours(cropped_image, good[2], -1, (0,0,255), 2)
            cv2.drawContours(cropped_image, good[3], -1, (255,255,255), 2)

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
                
                print("Центр робота", CRX, CRY)
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


