import cv2
import numpy as np
import math
from math import sqrt
import socket
import time 
from aruco import *

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

def get_robots_points(img):
    global CRX
    global CRY
    global NY
    global NX
    global camera_matrix_usb
    global dist_coefs_usb
    
    frame = img                     #используем камеру 
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
                
                # print("Центр робота", CRX, CRY)
        except: pass
        
        return CRX, CRY, NX, NY
    except: 
        # print("nan")
        return 0, 0, 0, 0
    

aruco_list_pre = {102: [585, 437, 96], 107: [100, 431, 168], 
                  343: [233, 387, 184], 
                  44: [368, 367, 43], 191: [467, 364, 72], 
                  215: [302, 334, 198], 
                  2: [527, 317, 45], 79: [209, 306, 77], 
                  55: [462, 280, -61], 143: [120, 265, 154], 
                  63: [390, 261, 267], 33: [303, 249, 80], 
                  229: [521, 204, 155], 97: [440, 185, 90], 
                  247: [269, 164, 1], 205: [369, 145, 180], 
                  115: [464, 126, 255], 118: [98, 118, 102], 
                  27: [557, 105, 74], 125: [269, 94, 145]}


def angle_between_points(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Вычисляем векторы между точками
    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)
    # Вычисляем скалярное произведение векторов
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Вычисляем длины векторов
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Вычисляем угол между векторами
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = math.acos(cos_angle)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def send_message(MESSAGE):
    UDP_IP = '10.128.73.102'
    UDP_PORT = 5005
    
    # print("UDP target IP: %s" % UDP_IP)
    # print("UDP target port: %s" % UDP_PORT)
    # print("message: %s" % MESSAGE)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP   
    sock.sendto(MESSAGE.encode('UTF-8'), (UDP_IP, UDP_PORT))
    
def vector_product_2d(v1, v2):
    """
    Вычисляет векторное произведение двух векторов в 2D.
    v1 и v2 - списки координат векторов.
    """
    return v1[0]*v2[1] - v1[1]*v2[0]

def point_side_2d(point1, point2, point_to_check):
    """
    Определяет, с какой стороны от прямой, образованной точками point1 и point2, лежит точка point_to_check.
    point1, point2, point_to_check - списки координат точек.
    """
    # Определяем векторы
    vector1 = [point2[0] - point1[0], point2[1] - point1[1]]
    vector2 = [point_to_check[0] - point1[0], point_to_check[1] - point1[1]]
    
    
    # print("Vector1", vector1)
    # print("Vector2", vector2)
    # Вычисляем векторное произведение
    cross_product = vector_product_2d(vector1, vector2)
    # print("cross_product", cross_product)
    # Проверяем знак вектора
    if cross_product > 0:
        return "3"
    elif cross_product < 0:
        return "4"
    else:
        return "0"

#получаю координаты центра робота и его направления 
def get_coord():
    ret, img = cam_usb.read()
    CRX2, CRY2, NX2, NY2 = get_robots_points(img)
    count_lol = 0
    while CRX2 == 0:
        time.sleep(0.3)
        if count_lol >= 10:
            return 0, 0, 0, 0
        CRX2, CRY2, NX2, NY2 = get_robots_points(img)
        count_lol+=1
    return CRX2, CRY2, NX2, NY2

# доезд к маркеру 
def to_mark(k = 1.0, ksm = 0.347):
    global x_mark
    global y_mark
    
    CRX2, CRY2, NX2, NY2 = get_coord()

    MESSAGE = "1" + "125" + "0" * (3 - len(str(int(distance_between_points((CRX2, CRY2), (x_mark, y_mark)) * ksm * k)))) + str(int(distance_between_points((CRX2, CRY2), (x_mark, y_mark)) * ksm * k))

    send_message(MESSAGE)
    
    CRX_last = -1
    while True:
        CRX_now, _, _, _ = get_coord()
        if CRX_last == CRX_now:    
            print("Robot has come", CRX_now, CRX_last)
            return
        else:
            time.sleep(1)
        CRX_last = CRX_now
        
    print("ads", CRX2, CRY2)

# поворот к маркеру
def turn_to_mark():
    global x_mark
    global y_mark
    CRX2, CRY2, NX2, NY2 = get_coord()

    angle = angle_between_points((NX2, NY2), (CRX2, CRY2), (x_mark, y_mark))

    MESSAGE = point_side_2d((CRX2, CRY2), (NX2, NY2), (x_mark, y_mark)) + "125" + "0" * (3 - len(str(int(round(angle, 0))))) + str(int(round(angle, 0))) 

    send_message(MESSAGE)
    
    CRX_last = -1
    while True:
        CRX_now, _, _, _ = get_coord()
        if CRX_last == CRX_now:    
            print("Robot has turned", "angle: ", angle, point_side_2d((CRX2, CRY2), (NX2, NY2), (x_mark, y_mark)),(CRX2, CRY2), (NX2, NY2), (x_mark, y_mark))
            return
        else:
            time.sleep(1)
        CRX_last = CRX_now
    
def mark_norm():
    global CRX
    global CRY
    global NY
    global NX
    global gr_mark
    
    ret, img = cam_usb.read()

    CRX, CRY, NX, NY = get_robots_points(img)

    high, _ = img.shape[:2]

    angle = angle_between_points((NX, NY), (CRX, CRY), (CRX, high))

    print("Угол для поворота по ориентации маркера:", int(round(angle, 0))) 

    MESSAGE = point_side_2d((CRX, CRY), (NX, NY), (CRX, high)) + "125" + "0" * (3 - len(str(int(round(angle, 0))))) + str(int(round(angle, 0)))
    if angle > 3:
        send_message(MESSAGE)
    gr_mark -= 180
    if gr_mark > 0:
        MESSAGE = "4" + "125" + "0" * (3 - len(str(int(round(abs(gr_mark), 0))))) + str(int(round(abs(gr_mark), 0)))
    else:
        MESSAGE = "3" + "125" + "0" * (3 - len(str(int(round(abs(gr_mark), 0))))) + str(int(round(abs(gr_mark), 0)))
        
    send_message(MESSAGE)
    
def go_to_marker_main(id_mark): 
    global CRX
    global CRY
    global NY
    global NX
    global aruco_list_upd
    global x_mark
    global y_mark
    global gr_mark
    
    x_mark = 0
    y_mark = 0
    gr_mark = 0
    
    if id_mark not in aruco_list_upd:
        gr_mark = aruco_list_pre[id_mark][2]
        x_mark = aruco_list_pre[id_mark][0]
        y_mark = aruco_list_pre[id_mark][1]
        print("aruco_list_pre is using")
    else:
        gr_mark = aruco_list_upd[id_mark][2]
        x_mark = aruco_list_upd[id_mark][0]
        y_mark = aruco_list_upd[id_mark][1]
        print("aruco_list_upd is using")

    _, img = cam_usb.read()
    CRX2, CRY2, _, _ = get_robots_points(img)
    # _, img = cam_usb.read()
    cv2.imwrite('n111' + '.png', img)
    print(CRX2, CRY2)
    
    turn_to_mark()
    to_mark(0.85)

    _, img = cam_usb.read()
    CRX2, CRY2, _, _ = get_robots_points(img)
    # _, img = cam_usb.read()
    cv2.imwrite('n112' + '.png', img)
    print(CRX2, CRY2)
    
    turn_to_mark()
    to_mark(1)
    
    _, img = cam_usb.read()
    CRX2, CRY2, _, _ = get_robots_points(img)
    # _, img = cam_usb.read()
    cv2.imwrite('n113' + '.png', img)
    print("Dist between robot's and point's centers", distance_between_points((CRX2, CRY2), (x_mark, y_mark)) * 0.347, distance_between_points((CRX2, CRY2), (x_mark, y_mark)),CRX2,CRY2,x_mark,y_mark)
    # if distance_between_points((CRX2, CRY2), (x_mark, y_mark)) * 0.347 > 15:
        # turn_to_mark()
        # to_mark(1)
    # send_message("1125010")
    # turn_to_mark()
    # to_mark(1)
    mark_norm()
    send_message("3125360")


cam_usb = cv2.VideoCapture(0)
cam_usb.set(cv2.CAP_PROP_AUTO_WB, 0.0) 

#прогрев камеры
for _ in range(30):
    _, img = cam_usb.read()
    
aruco_list_upd = get_aruco_list(img)
print(aruco_list_upd)

CRX, CRY = 0, 0 
NX, NY = 0, 0


camera_matrix_usb = np.array([[744.78560704,   0.        , 416.79106153],
       [  0.        , 746.17464309, 310.35201284],
       [  0.        ,   0.        ,   1.        ]])
dist_coefs_usb = np. array([-0.01179911,  0.19509182,  0.00176673,  0.00154678, -0.42737765])

go_to_marker_main(205)
# CRX, CRY = 0, 0 
# NX, NY = 0, 0
# go_to_marker_main(55)
# CRX, CRY = 0, 0 
# NX, NY = 0, 0
# go_to_marker_main(44)
# CRX, CRY = 0, 0 
# NX, NY = 0, 0
# print("next")
# time.sleep(2)

# go_to_marker_main(55)
# print("next")
# time.sleep(2)
# go_to_marker_main(44)

