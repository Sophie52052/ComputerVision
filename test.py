#基本人臉特徵
#dlib
import dlib
from PIL import Image
import numpy as np
import cv2
 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img_cv2 = cv2.imread("./images/Trump.jpg")

img_PIL = Image.open("./images/Trump.jpg").convert('RGBA')

img_gray = np.array(img_PIL.convert('L'))

rects = detector(img_gray, 0)

for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_cv2, rects[i]).parts()])
    img_cv2 = img_cv2.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        # 利用cv2.circle給每個特徵點畫一個圈，共68個
        cv2.circle(img_cv2, pos, 5, color=(0, 255, 0))
        # 利用cv2.putText輸出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_cv2, str(idx+1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        
cv2.namedWindow("img_cv2", 2)
cv2.imshow("img_cv2", img_cv2)
cv2.waitKey(0)