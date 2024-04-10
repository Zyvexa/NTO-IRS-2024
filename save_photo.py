import cv2

# Включаем первую камеру
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
# "Прогреваем" камеру, чтобы снимок не был тёмным
#for i in range(30):
 #   cap.read()
#ret, frame = cap.read()
for i in range(20):
# Делаем снимок
    ret, frame = cap.read()
    cv2.imshow('img', frame)
    cv2.waitKey()
    # Записываем в файл
    cv2.imwrite('n' + str(i) + '.png', frame)

# Отключаем камеру
cap.release()
